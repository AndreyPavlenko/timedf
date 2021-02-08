USER admin omnisci {

DROP TABLE IF EXISTS trips_reduced;
CREATE TABLE trips_reduced (
    pickup_datetime         TIMESTAMP,
    passenger_count         SMALLINT,
    trip_distance           DECIMAL(14,2),
    total_amount            DECIMAL(14,2),
    cab_type                VARCHAR(6) ENCODING DICT
) WITH (FRAGMENT_SIZE=20000000);
--COPY trips_reduced FROM 'trips_reduced-full.csv' WITH (header='true');
}
