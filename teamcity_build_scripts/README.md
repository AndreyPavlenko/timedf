1. TeamCity host should have OmnisSciDB dependencies installed. Either
by using `scripts/mapd-deps-ubuntu.sh` ([see here for other
distros](https://github.com/omnisci/omniscidb/wiki/OmniSciDB-Dependencies))
or downloading and installing binaries from omnisci site.
2. User should have high limit for open files. Using `ulimit -n 10000`
in `~/.bashrc` is a good idea.
3. The following python3 packages should be installed using `pip`:
pymapd, braceexpand, mysql-connector-python.
4. User should have permissions to connect to MySQL host and insert
records there. Modify test scripts with user credentials and MySQL
server host name.
