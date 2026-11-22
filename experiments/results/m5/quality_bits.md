# What the replay QUALITY flags mean (plan 4.4, for R9)

Checked 2026-11-20 against JSOC's own documentation. Inputs only: no label or model output was read.

## Values seen in replay

From the M4 inputs check (2,869 of 5,101 replay records, 56.2%, have QUALITY != 0): `0x0` 2,232, `0x80` 1,421,
`0x10080` 1,392 (M4 work report 10.10).

## Bit meanings (HMI level 1.5 and derived products, "created but in sub-optimal conditions", QUALITY > 0)

Source: `http://jsoc.stanford.edu/doc/data/hmi/Quality_Bits/QUALITY.txt` (file dated 2021-03-04, header "updated on
May 24, 2013") and `QUALITY.notes` in the same folder (dated 2017-09-19).

| Bit | Name | JSOC text |
| --- | --- | --- |
| `0x00000080` | `QUAL_TEMPERROR` | "Code error discovered, will be corrected in later versions" |
| `0x00010000` | `QUAL_LOWINTERPNUM` | "the number of averaging points is lower than TempIntNum, AND/OR 2 interpolation points were separated by more than the cadence" |
| `0x00000400` | `QUAL_NOCOSMICRAY` | "some cosmic-ray hit lists could not be read" (with `0x10000` this is the daily `0x00010400` flag at 06:00/18:00 TAI in 2011-15, `configs/data.yaml` `sampling`) |

So `0x10080` = `QUAL_LOWINTERPNUM` + `QUAL_TEMPERROR`.

`QUALITY.notes` on bit `0x80`: "In cases where 45s or 720s data exists and there is a code error detected we will set
bit 0x80 for existing records while they are waiting replacement with reprocessed data. [...] Sept 2017. for CALVER64
masked with 0xF0000 leaves 0x20000 for the initial year+ of modL data, we set QUALITY 0x80 while waiting for
reprocessing of hmi.S_720s and all of its derived products. The records with this QUALITY bit set should not be
visible once the reprocessing is done."

## What this means for R9

- The flag marks an **HMI calibration version waiting for reprocessing** (the first year or so of the "mod L"
  calibration), set on otherwise usable records. It is not a missing or corrupt image.
- The flagged months in replay (2016-04 through 2017-02, nearly every record; plus 2017-03 and 2017-08) match "the
  initial year+ of modL data".
- Risk for the model: a calibration difference shifts Br values slightly. That is the R7/R9 input-shift risk, and
  it is what monitoring signal S2 (median non-zero |Br| vs the training reference) and the `quality_nonzero` slice
  look at.

## Spot check of the calibration version (free JSOC keyword query, 7 records, 2026-11-20 23:0x)

| HARPNUM | T_REC | QUALITY | CALVER64 | CALVER64 & 0xF0000 |
| --- | --- | --- | --- | --- |
| 6483 | 2016.04.14_00:00:00_TAI | 0x80 | 0x42012 | 0x40000 |
| 6489 | 2016.04.14_00:00:00_TAI | 0x80 | 0x42012 | 0x40000 |
| 6483 | 2016.04.14_06:00:00_TAI | 0x10080 | 0x42012 | 0x40000 |
| 6489 | 2016.04.14_06:00:00_TAI | 0x10080 | 0x42012 | 0x40000 |
| 6281 | 2016.01.28_12:00:00_TAI | 0x0 | 0x2012 | 0x0 |
| 6289 | 2016.01.28_12:00:00_TAI | 0x0 | 0x2012 | 0x0 |
| 7034 | 2017.06.02_18:12:00_TAI | 0x0 | 0x42012 | 0x40000 |

**Unverified:** the flagged records carry the same calibration version (0x4xxxx) as an unflagged June 2017 record,
not the 0x20000 the note names. Whether these SHARP records were reprocessed and kept the bit, or are still the
pre-reprocessing version, cannot be settled from 7 records. The test treats the flag as the plan says: the clean
subset is primary, all records and flagged records are reported alongside.
