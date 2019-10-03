package main

import (
	"bufio"
	"flag"
	"fmt"
	"io"
	"log"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"text/template"
)

const (
	omnisciExecutable  = "build/bin/omnisql"
	taxyTripsDirectory = "/localdisk/work/trips_x*.csv"

	command1DropTableTrips = "drop table taxitestdb;"
	command2ImportCSV      = "COPY taxitestdb FROM '%s' WITH (header='false');"

	timingRegexpString    = `Execution time: (\d+) ms, Total time: (\d+) ms`
	exceptionRegexpString = `Exception: .*`
)

var (
	omnisciCmdLine = []string{"-q", "omnisci", "-u", "admin", "-p", "HyperInteractive"}
	benchmarksCode = []string{
		`\timing
SELECT cab_type,
       count(*)
FROM taxitestdb
GROUP BY cab_type;
`,
		`\timing
SELECT passenger_count,
       avg(total_amount)
FROM taxitestdb
GROUP BY passenger_count;
`,
		`\timing
SELECT passenger_count,
       extract(year from pickup_datetime) AS pickup_year,
       count(*)
FROM taxitestdb
GROUP BY passenger_count,
         pickup_year;
`,
		`\timing
SELECT passenger_count,
       extract(year from pickup_datetime) AS pickup_year,
       cast(trip_distance as int) AS distance,
       count(*) AS the_count
FROM taxitestdb
GROUP BY passenger_count,
         pickup_year,
         distance
ORDER BY pickup_year,
         the_count desc;
`,
	}
	timingRegexp    = regexp.MustCompile(timingRegexpString)
	exceptionRegexp = regexp.MustCompile(exceptionRegexpString)
)

type arrayFlags []uint64

func (af *arrayFlags) String() string {
	return fmt.Sprint(*af)
}

func (af *arrayFlags) Set(value string) error {
	ival, err := strconv.ParseUint(value, 10, 64)
	if err != nil {
		return err
	}
	*af = append(*af, ival)
	return nil
}

var myFlags arrayFlags

func main() {
	var fragmentSizes arrayFlags
	flag.Var(&fragmentSizes, "fs", "Fragment size to use for created table. Multiple values are allowed and encouraged.")
	datafiles := flag.Uint("df", 1, "Number of datafiles to input into database for processing")
        datafilesPattern := flag.String("dp", taxyTripsDirectory, "Wildcard pattern of datafiles that should be loaded")
	dnd := flag.Bool("dnd", false, "Do not delete old table. KEEP IN MIND that in this case -fs values have no effect because table is taken from previous runs.")
	dni := flag.Bool("dni", false, "Do not create new table and import any data from CSV files. KEEP IN MIND that in this case -fs values have no effect because table is taken from previous runs.")
	times := flag.Uint("t", 5, "Number of times to run every benchmark. Best result is selected")
	showOut := flag.Bool("sco", false, "Hide commands output")
	showBench := flag.Bool("sbo", false, "Show benchmarks output")
	repFile := flag.String("r", "report.csv", "Report file name")
	testme := flag.Bool("test", false, "Run tests")
	serverPort := flag.Int("port", 62074, "TCP port to use to connect to server")
	flag.Parse()

	if *testme {
		test()
	}

	if *datafiles <= 0 {
		log.Fatal("Bad number of data files specified", *datafiles)
	}
	if *times < 1 {
		log.Fatal("Bad number of iterations specified", *times)
	}

	fileOut, err := os.Create(*repFile)
	if err != nil {
		log.Fatal(err)
	}

	omnisciCmdLine = append(omnisciCmdLine, "--port", strconv.Itoa(*serverPort))

	tripsTmpl := template.Must(template.ParseFiles("trips.tmpl"))

	for _, fs := range fragmentSizes {
		fmt.Println("RUNNING WITH FRAGMENT SIZE", fs)
		// Delete old table
		if !*dnd {
			fmt.Println("Deleting taxitestdb old database")
			cmd := exec.Command(omnisciExecutable, omnisciCmdLine...)
			cmd.Stdin = strings.NewReader(command1DropTableTrips)
			output, err := cmd.CombinedOutput()
			if err != nil {
				log.Fatal(err)
			}
			if *showOut {
				fmt.Printf("%s", output)
			}
			fmt.Println("Command returned", cmd.ProcessState.ExitCode())
		}

		// Data files import
		if !*dni {
			// Create new table
			fmt.Println("Creating new table taxitestdb with fragment size", fs)
			cmd := exec.Command(omnisciExecutable, omnisciCmdLine...)
			pipeReader, pipeWriter := io.Pipe()
			cmd.Stdin = pipeReader
			go func() {
				tripsTmpl.ExecuteTemplate(pipeWriter, "trips", fs)
				pipeWriter.Close()
			}()
			output, err := cmd.CombinedOutput()
			if err != nil {
				log.Fatal(err)
			}
			if !*showOut {
				fmt.Printf("%s", output)
			}
			fmt.Println("Command returned", cmd.ProcessState.ExitCode())

			dataFileNames, err := filepath.Glob(*datafilesPattern)
			if err != nil {
				log.Fatal(err)
			} else if len(dataFileNames) == 0 {
				log.Fatal("Could not find any data files matching", taxyTripsDirectory)
			}
			if *datafiles > uint(len(dataFileNames)) {
				*datafiles = uint(len(dataFileNames))
			}
			for df := uint(0); df < *datafiles; df++ {
				fmt.Println("Importing datafile", dataFileNames[df])
				cmd := exec.Command(omnisciExecutable, omnisciCmdLine...)
				cmdString := fmt.Sprintf(command2ImportCSV, dataFileNames[df])
				cmd.Stdin = strings.NewReader(cmdString)
				output, err := cmd.CombinedOutput()
				if err != nil {
					log.Fatal(err)
				}
				if !*showOut {
					fmt.Printf("%s", output)
				}
				fmt.Println("Command returned", cmd.ProcessState.ExitCode())
			}
		}

		// Benchmarks
		for benchNumber, benchString := range benchmarksCode {
			bestExecTime := uint64(math.MaxUint64)
			bestTotalTime := uint64(math.MaxUint64)
			errstr := ""
			for iter := uint(1); iter <= *times; iter++ {
				fmt.Println("Running benchmark number", benchNumber+1, "Iteration number", iter)
				cmd := exec.Command(omnisciExecutable, omnisciCmdLine...)
				cmd.Stdin = strings.NewReader(benchString)
				output, err := cmd.CombinedOutput()
				if err != nil {
					log.Fatal(err)
				}
				if *showBench {
					fmt.Printf("%s", output)
				}
				fmt.Println("Command returned", cmd.ProcessState.ExitCode())
				strout := string(output)
				matches := timingRegexp.FindStringSubmatch(strout)
				execTime := uint64(math.MaxUint64)
				totalTime := uint64(math.MaxUint64)
				if len(matches) == 3 {
					errHandle := func(n int, err error, str string) {
						if err != nil {
							log.Fatal(err)
						} else if n != 1 {
							errstr = fmt.Sprintf("Failed to parse number %s", str)
							fmt.Println(errstr)
						}
					}
					n, err := fmt.Sscanf(matches[1], "%d", &execTime)
					errHandle(n, err, matches[1])
					n, err = fmt.Sscanf(matches[2], "%d", &totalTime)
					errHandle(n, err, matches[2])
				} else {
					fmt.Printf("Failed to parse command output: %s\n", output)
					errstr = getErrorLine(strout)
				}
				fmt.Printf("Command exec time %d, total time %d\n", execTime, totalTime)
				if execTime < bestExecTime {
					bestExecTime = execTime
				}
				if totalTime < bestTotalTime {
					bestTotalTime = totalTime
				}
			}
			fmt.Printf("BENCHMARK %d exec time %d, total time %d\n", benchNumber+1, bestExecTime, bestTotalTime)
			fmt.Fprintf(fileOut, "%d,%d,%d,%d,%d,%s\n", *datafiles, fs, benchNumber+1, bestExecTime, bestTotalTime, errstr)
		}
	}
	fileOut.Close()
}

func getErrorLine(buffer string) string {
	matches := exceptionRegexp.FindStringSubmatch(buffer)
	if len(matches) == 1 {
		return matches[0]
	}

	scanner := bufio.NewScanner(strings.NewReader(buffer))
	var str string
	for scanner.Scan() {
		if len(scanner.Text()) > 0 {
			str = scanner.Text()
		}
	}
	return str
}

func test() {
	for benchNumber, benchString := range benchmarksCode {
		fmt.Println(benchNumber, ":", getErrorLine(benchString))
	}

	teststr1 := `this is a test
Exception: Sorting the result would be too slow
this is a test
`
	teststr2 := `this is a test1
this is a test2

`
	fmt.Println("test1:", getErrorLine(teststr1))
	fmt.Println("test2:", getErrorLine(teststr2))
}
