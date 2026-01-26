from __future__ import annotations

from typing import List

def _sql_quote(val: str)-> str:
    return str(val).replace("'","''")

def build_positions_flights_lookup_sql(reg: str, dep_iata: str, arr_iata: str, date_part: str)-> str:
    regq = _sql_quote(reg)
    depq = _sql_quote(dep_iata)
    arrq = _sql_quote(arr_iata)
    dp = _sql_quote(date_part)
    return f"""SELECT FLIGHT_ID
FROM flightradarrepository.positions_flights
WHERE REG = '{regq}'
  AND schd_from = '{depq}'
  AND schd_to = '{arrq}'
  AND date_part = '{dp}'""".strip()

def _format_in_list(values:List)-> str:
    esc = [_sql_quote(str(v)) for v in values if v is not None]
    return "(" + ", ".join(f"'{x}'" for x in esc) + ")"

SELECT_TEMPLATE = """SELECT
  T1.flight_id AS fr24_flight_id,
  T1.reg AS fr24_registration,
  T1.equip AS fr24_equipment,
  T1.callsign AS fr24_callsign,
  T1.flight AS fr24_flight,
  T1.schd_from AS fr24_departure,
  T1.schd_to AS fr24_arrival,
  T1.real_to AS fr24_real_arrival,
  T1.date_part AS fr24_positionsflights_datepartition,
  from_unixtime(T2.snapshot_id) AS fr24_obs_timestamp,
  date(from_unixtime(T2.snapshot_id)) AS fr24_obs_date,
  date_format(from_unixtime(T2.snapshot_id), '%H:%i:%s') AS fr24_obs_time,
  T2.altitude as fr24_altitude,
  T2.heading as fr24_heading,
  T2.latitude as fr24_latitude,
  T2.longitude as fr24_longitude,
  T2.speed as fr24_speed,
  T2.dt_partition as fr24_positions_datepartition
FROM flightradarrepository.positions_flights T1
LEFT JOIN flightradarrepository.positions T2
  ON CAST(T1.FLIGHT_ID as varchar) = T2.FLIGHT_ID AND T1.date_part = T2.dt_partition
WHERE {where_clause}
ORDER BY T2.snapshot_id ASC""".strip()

def build_positions_sql(flight_ids: List, date_part: str)-> str:
    dp = _sql_quote(str(date_part))
    in_list = _format_in_list(flight_ids)
    where_clause = f"T1.date_part = '{dp}' AND CAST(T1.FLIGHT_ID as varchar) IN {in_list}"
    return SELECT_TEMPLATE.format(where_clause=where_clause)