import pandas as pd
from mtx.db.db_GIS import get_devices_with_structures

import pyodbc
import os
import pandas as pd

# SQL credentials are supplied by the Windows/user environment.
USERNAME = os.getenv("SMARTLINK_AMI_USERNAME", "asset_ro")
PASSWORD = os.getenv("SMARTLINK_AMI_PASSWORD")

def get_connection(server: str, database: str):
    if not PASSWORD:
        raise RuntimeError("Set SMARTLINK_AMI_PASSWORD before querying AMI.")
    conn_str = (
        "Driver={SQL Server};"
        f"Server={server};"
        f"Database={database};"
        f"UID={USERNAME};"
        f"PWD={PASSWORD};"
        "TrustServerCertificate=yes;"
    )
    return pyodbc.connect(conn_str)

def run_query_ami(query: str, server: str, database: str) -> pd.DataFrame:
    with get_connection(server, database) as conn:
        return pd.read_sql_query(query, conn)

def run_query_with_temp_ami(query_setup: str, query_final: str, server: str, database: str) -> pd.DataFrame:
    conn = get_connection(server, database)
    cur = conn.cursor()
    
    # Run setup query (e.g., create temp tables)
    cur.execute("SET NOCOUNT ON; " + query_setup)
    while cur.nextset():
        pass  # Advance through all intermediate results

    # Run final query and load results into DataFrame
    df = pd.read_sql_query(query_final, conn)

    # Clean up
    cur.close()
    conn.close()
    
# ======================================================
# CONNECTIONS
# ======================================================
SERVER = "ct-omsdbprod.osg.avangrid.net"
CMP_DATABASE = "AMI_SDE"
NY_DATABASE = "NYAMI_SDE"


# ======================================================
# AMI DATA
# ======================================================
def get_cmp_ami_data():
    query = """
        SELECT
            c1.MeterID,
            c2.Circuit,
            c1.METERFUNCTIONALLOC AS SuperiorFLOC,
            shape.STX AS LONGITUDE,
            shape.STY AS LATITUDE,
            SAP_Address,
            SAP_Town,
            ZIP,
            SAP_Address + ', ' + SAP_Town + ', ' + ZIP AS ADDRESS,

            CASE
                WHEN TRY_CAST(COALESCE(c1.SERVICECENTER, c2.SERVICECENTER) AS INT) = 79 THEN 'Brunswick'
                WHEN TRY_CAST(COALESCE(c1.SERVICECENTER, c2.SERVICECENTER) AS INT) = 80 THEN 'Penobscot/Rockland'
                WHEN TRY_CAST(COALESCE(c1.SERVICECENTER, c2.SERVICECENTER) AS INT) = 81 THEN 'Alfred'
                WHEN TRY_CAST(COALESCE(c1.SERVICECENTER, c2.SERVICECENTER) AS INT) = 82 THEN 'Portland'
                WHEN TRY_CAST(COALESCE(c1.SERVICECENTER, c2.SERVICECENTER) AS INT) = 83 THEN 'Augusta'
                WHEN TRY_CAST(COALESCE(c1.SERVICECENTER, c2.SERVICECENTER) AS INT) = 84 THEN 'Fairfield'
                WHEN TRY_CAST(COALESCE(c1.SERVICECENTER, c2.SERVICECENTER) AS INT) = 85 THEN 'Lewiston'
                WHEN TRY_CAST(COALESCE(c1.SERVICECENTER, c2.SERVICECENTER) AS INT) = 86 THEN 'Farmington'
                WHEN TRY_CAST(COALESCE(c1.SERVICECENTER, c2.SERVICECENTER) AS INT) = 87 THEN 'Dover'
                WHEN TRY_CAST(COALESCE(c1.SERVICECENTER, c2.SERVICECENTER) AS INT) = 88 THEN 'Skowhegan'
                WHEN TRY_CAST(COALESCE(c1.SERVICECENTER, c2.SERVICECENTER) AS INT) = 89 THEN 'Bridgton'
                ELSE 'Unknown Division Code'
            END AS Division,

            'CMP' AS OPCO

        FROM [AMI_SDE].[AMI].CRMBCustomers c1
        INNER JOIN [AMI_SDE].[AMI].[ALL_METERS_CRMB] c2
            ON c2.MeterID = c1.METERID
    """

    return run_query_ami(query, SERVER, CMP_DATABASE)


def get_ny_ami_data():
    query = """
         SELECT
            s.MeterID,
            s.PremiseType,
			s.PremiseID,
            s.ConnectionObject,
            s.Circuit,
			s.City,
			s.ZipCode,
            s.ServiceCenterName AS Division,
            c.shape.STX AS LONGITUDE,
            c.shape.STY AS LATITUDE,
            s.AMIMeter,
			c.Address,
            s.SuperiorFLOC,
			CONCAT(c.Address, ', ', s.City, ', ', s.ZipCode) AS FullAddress,
            CASE
                WHEN s.SuperiorFLOC LIKE '%9301%' THEN 'NYSEG'
                WHEN s.SuperiorFLOC LIKE '%9302%' THEN 'RGE'
                ELSE NULL
            END AS OPCO

        FROM [NYAMI_SDE].[NYAMI].[SAPCRMBCustomer] s
        INNER JOIN [NYAMI_SDE].[NYAMI].[ALL_METERS_NY] c
            ON c.MeterID = s.MeterID
        WHERE s.ServiceType = 'EL'

          AND s.AMIMeter = 'Y'
    """

    return run_query_ami(query, SERVER, NY_DATABASE)


def get_all_ami_data():
    return pd.concat(
        [
            get_cmp_ami_data(),
            get_ny_ami_data(),
        ],
        ignore_index=True,
    )


# ======================================================
# ADAM VALIDATED DATA
# ======================================================

def _database_identifier(series):
    if pd.api.types.is_numeric_dtype(series) and series.dropna().mod(1).eq(0).all():
        series = series.astype("Int64")
    clean = series.astype("string").str.strip()
    return clean.mask(clean.eq(""))


def get_adam_validated_data():
    query = """
        WITH Deduped AS (
            SELECT
                f.MeterNumber,
                f.Validated_Trans_GISID AS Truth,
                t2.ServiceCenterName AS Division,
                t2.SuperiorFLOC AS MeterFLOC,
                f.ValidationSource,
                ROW_NUMBER() OVER (
                    PARTITION BY f.MeterNumber, f.Validated_Trans_GISID
                    ORDER BY f.ValidationSource
                ) AS rn
            FROM [NYAMI_SDE].[NYAMI].[TA_METER_TRANS_VERIFIED] f
            LEFT JOIN [NYAMI_SDE].[NYAMI].[SAPCRMBCustomer] t2
                ON t2.MeterID = f.MeterNumber
        )
        SELECT
            MeterNumber,
            Truth,
            Division,
            MeterFLOC,
            ValidationSource
        FROM Deduped
        WHERE rn = 1
    """

    transformer_data = get_devices_with_structures(
        server="nygisprd19",
        database="NYGIS",
        device="transformer",
    )

    df = run_query_ami(query, SERVER, NY_DATABASE)

    df = df.rename(columns={"MeterNumber": "meter_number"}).copy()
    for column in ("meter_number", "Truth"):
        df[column] = _database_identifier(df[column])
    lookup = transformer_data[["FuncLoc", "GISID"]].copy()
    for column in lookup:
        lookup[column] = _database_identifier(lookup[column])
    lookup = lookup.dropna(subset=["GISID", "FuncLoc"]).drop_duplicates()
    if lookup["GISID"].duplicated().any():
        raise ValueError("Verified GIS transformer IDs map to multiple FLOCs.")
    df = df.dropna(subset=["meter_number"]).drop_duplicates()
    df = df.merge(lookup, left_on="Truth", right_on="GISID", how="left", validate="many_to_one")
    if (df["Truth"].notna() & df["FuncLoc"].isna()).any():
        raise ValueError("Verified transformer GIS IDs are missing from the current transformer reference; resolve before ADAM history.")
    df = (df.drop(columns=["Truth", "GISID"])
          .rename(columns={"FuncLoc": "Correct_Transformer_FLOC"})
          .reset_index(drop=True))

    return df


# ======================================================
# ITRON VALIDATION INFO
# ======================================================
def get_itron_validation_info():
    query = """
        WITH Deduped AS (
            SELECT
                f.MeterNumber,
                f.Validated_Trans_GISID AS Truth,
                t2.ServiceCenterName AS Division,
                t2.SuperiorFLOC AS MeterFLOC,
                f.ValidationSource,
                t3.Tx_Subgroup_ID,
                t3.Tx_Subgroup_GIS_Correlation,
                t3.SPs_in_Tx_Subgroup,
                ROW_NUMBER() OVER (
                    PARTITION BY f.MeterNumber, f.Validated_Trans_GISID
                    ORDER BY f.ValidationSource
                ) AS rn
            FROM [NYAMI_SDE].[NYAMI].[TA_METER_TRANS_VERIFIED] f
            LEFT JOIN [NYAMI_SDE].[NYAMI].[SAPCRMBCustomer] t2
                ON t2.MeterID = f.MeterNumber
            LEFT JOIN [NYAMI_SDE].[NYAMI].[TA_METER] t3
                ON t3.Meter_ID = f.MeterNumber
        )
        SELECT *
        FROM Deduped
        WHERE rn = 1
    """

    return run_query_ami(query, SERVER, NY_DATABASE)
# ======================================================
# C2T REVIEWED BY ADAM
# ======================================================
def get_adam_c2t():
    query = """
    select meter_number, validation_linetype,ValidationSource, Validated_Trans_GISID, c.model_recommened_floc, c.model_recommened_TXID,
    funcloc
    from TA_METER_TRANS_VERIFIED 
    inner join c2t_analysis c on meter_number = meternumber 
    inner join DSTRU s on [model_recommened_floc] = funcloc inner join dxfr x on s.Z_GISID = x.z_poleid
    where ValidationSource in ('MAP') 
    """

    return run_query_ami(query, SERVER, NY_DATABASE)


# ======================================================
# ITRON METER DATA
# ======================================================
def get_itron_meter_data(meter_ids=None):
    base_query = """

        SELECT
            t.Meter_Number AS meter_number,
            t.LA_Transformer_ID,
			t.GIS_Transformer_ID,
            t.Nearest_Transformer_ID,
            t.Tx_Subgroup_ID,
            t.Tx_Subgroup_GIS_Correlation,
            t.SPs_in_Tx_Subgroup,

            s.SuperiorFLOC AS current_meter_floc,
            s.ServiceCenterName AS division,
            s.Circuit,
            s.PremiseType,
            s.ConnectionObject,
            s.City,
            s.COCity,
            s.ZipCode,

            CASE
                WHEN s.SuperiorFLOC LIKE '%9301%' THEN 'NYSEG'
                WHEN s.SuperiorFLOC LIKE '%9302%' THEN 'RGE'
                ELSE NULL
            END AS opco,

            m.SHAPE.STX AS longitude,
            m.SHAPE.STY AS latitude

        FROM [NYAMI_SDE].[NYAMI].[TA_METER] t
        LEFT JOIN [NYAMI_SDE].[NYAMI].[SAPCRMBCustomer] s
            ON s.MeterID = t.Meter_Number
        LEFT JOIN [NYAMI_SDE].[NYAMI].[ALL_METERS_NY] m
            ON m.MeterID = t.Meter_Number
        WHERE s.ServiceType = 'EL'
          AND s.AMIMeter = 'Y'
    """

    if meter_ids is None:
        df = run_query_ami(base_query, SERVER, NY_DATABASE)

        return (
            df
            .drop_duplicates()
            .reset_index(drop=True)
        )

    meter_ids = (
        pd.Series(meter_ids)
        .dropna()
        .astype(str)
        .drop_duplicates()
        .tolist()
    )

    results = []

    for i in range(0, len(meter_ids), 500):
        chunk = meter_ids[i : i + 500]
        meter_sql = ",".join(f"'{m}'" for m in chunk)

        query = f"""
            {base_query}
            AND t.Meter_ID IN ({meter_sql})
        """

        results.append(
            run_query_ami(query, SERVER, NY_DATABASE)
        )

    return (
        pd.concat(results, ignore_index=True)
        .drop_duplicates()
        .reset_index(drop=True)
    )
