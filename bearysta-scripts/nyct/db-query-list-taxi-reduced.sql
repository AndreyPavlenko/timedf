USER admin omnisci {

SELECT cab_type, count(*) FROM trips_reduced GROUP BY cab_type;
SELECT cab_type, count(*) FROM trips_reduced GROUP BY cab_type;
SELECT cab_type, count(*) FROM trips_reduced GROUP BY cab_type;

SELECT passenger_count, avg(total_amount) FROM trips_reduced GROUP BY passenger_count;
SELECT passenger_count, avg(total_amount) FROM trips_reduced GROUP BY passenger_count;
SELECT passenger_count, avg(total_amount) FROM trips_reduced GROUP BY passenger_count;

SELECT passenger_count, extract(year from pickup_datetime) AS pickup_year, count(*) FROM trips_reduced GROUP BY passenger_count, pickup_year;
SELECT passenger_count, extract(year from pickup_datetime) AS pickup_year, count(*) FROM trips_reduced GROUP BY passenger_count, pickup_year;
SELECT passenger_count, extract(year from pickup_datetime) AS pickup_year, count(*) FROM trips_reduced GROUP BY passenger_count, pickup_year;

SELECT passenger_count, extract(year from pickup_datetime) AS pickup_year, cast(trip_distance as int) AS distance, count(*) AS the_count FROM trips_reduced GROUP BY passenger_count, pickup_year, distance ORDER BY pickup_year, the_count desc;
SELECT passenger_count, extract(year from pickup_datetime) AS pickup_year, cast(trip_distance as int) AS distance, count(*) AS the_count FROM trips_reduced GROUP BY passenger_count, pickup_year, distance ORDER BY pickup_year, the_count desc;
SELECT passenger_count, extract(year from pickup_datetime) AS pickup_year, cast(trip_distance as int) AS distance, count(*) AS the_count FROM trips_reduced GROUP BY passenger_count, pickup_year, distance ORDER BY pickup_year, the_count desc;

SELECT cab_type, count(*) FROM trips_reduced GROUP BY cab_type;
SELECT cab_type, count(*) FROM trips_reduced GROUP BY cab_type;
SELECT cab_type, count(*) FROM trips_reduced GROUP BY cab_type;

SELECT passenger_count, avg(total_amount) FROM trips_reduced GROUP BY passenger_count;
SELECT passenger_count, avg(total_amount) FROM trips_reduced GROUP BY passenger_count;
SELECT passenger_count, avg(total_amount) FROM trips_reduced GROUP BY passenger_count;

SELECT passenger_count, extract(year from pickup_datetime) AS pickup_year, count(*) FROM trips_reduced GROUP BY passenger_count, pickup_year;
SELECT passenger_count, extract(year from pickup_datetime) AS pickup_year, count(*) FROM trips_reduced GROUP BY passenger_count, pickup_year;
SELECT passenger_count, extract(year from pickup_datetime) AS pickup_year, count(*) FROM trips_reduced GROUP BY passenger_count, pickup_year;

SELECT passenger_count, extract(year from pickup_datetime) AS pickup_year, cast(trip_distance as int) AS distance, count(*) AS the_count FROM trips_reduced GROUP BY passenger_count, pickup_year, distance ORDER BY pickup_year, the_count desc;
SELECT passenger_count, extract(year from pickup_datetime) AS pickup_year, cast(trip_distance as int) AS distance, count(*) AS the_count FROM trips_reduced GROUP BY passenger_count, pickup_year, distance ORDER BY pickup_year, the_count desc;
SELECT passenger_count, extract(year from pickup_datetime) AS pickup_year, cast(trip_distance as int) AS distance, count(*) AS the_count FROM trips_reduced GROUP BY passenger_count, pickup_year, distance ORDER BY pickup_year, the_count desc;

}
