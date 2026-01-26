from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict

@dataclass

class TaQueryParams:
    dt_start_utc: Optional[str] = None
    dt_end_utc: Optional[str] = None
    date_yyyy_mm_dd: Optional[str] = None
    airline: Optional[str] = None
    callsign: Optional[str] = None
    registration: Optional[str] = None
    departure_aerodome: Optional[str] = None
    destination_aerodome: Optional[str] = None

def build_ta_sql(params: TaQueryParams)->str:
    where = []

    def eq(col: str, val: Optional[str]):
        if val is None or str(val).strip()=="":
            return
        v = str(val).replace("'","''")
        where.append(f"{col} = '{v}'")

    eq("flight.airline", params.airline)
    eq("flight.callsign", params.callsign)
    eq("flight.registration", params.registration)
    eq("flight.departureaerodrome", params.departure_aerodome)
    eq("flight.destinationaerodrome", params.destination_aerodome)

    if params.dt_start_utc and params.dt_end_utc:
        s = params.dt_start_utc.replace("'","''")
        e = params.dt_end_utc.replace("'","''")
        where.append(
            "from_iso8601_timestamp(measurement.observationtime) "
            f"BETWEEN from_iso8601_timestamp('{s}') AND from_iso8601_timestamp('{e}')"
        )

    elif params.date_yyyy_mm_dd:
        d = params.date_yyyy_mm_dd.replace("'","''")
        where.append(f"date(from_iso8601_timestamp(measurement.observationtime)) = date '{d}'")
    
    where_clause = "\n AND ".join(where) if where else "1=1"

    sql = f"""
    SELECT
    measurement.observationtime AS observation_timestamp,
    date(from_iso8601_timestamp(measurement.observationtime)) AS observation_date,
    date_format(from_iso8601_timestamp(measurement.observationtime), '%H:%i:%s') AS observation_time,
    measurement.altitude AS altitude,
    measurement.latitude AS latitude,
    measurement.longitude AS longitude,
    measurement.temperature AS temperature,
    measurement.wind.speed AS wind_speed,
    measurement.wind.direction AS wind_direction,
    measurement.edr.algorithm AS algorithm,
    measurement.edr.peak.value AS peak_edr,
    measurement.edr.mean.value AS mean_edr,
    measurement.edr.ncar.meanconfidence AS mean_confidence,
    measurement.edr.ncar.peakconfidence AS peak_confidence,
    measurement.edr.ncar.numbergoodpoints AS number_good_points,
    measurement.edr.ncar.peaklocation AS peak_location,
    metadata.id AS id,
    metadata.tafi AS tafi,
    flight.airline AS airline,
    flight.callsign AS callsign,
    flight.registration AS registration,
    flight.departureaerodrome AS departure_aerodome,
    flight.destinationaerodrome AS destination_aerodome,
    date_hour AS date_hour
    FROM metrepository.measurements_json
    WHERE
    {where_clause}
    ORDER BY measurement.observationtime ASC
    """
    return sql.strip()
def params_to_dict(params: TaQueryParams)-> Dict:
    return params.__dict__.copy()