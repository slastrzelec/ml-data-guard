# Data

No data file is tracked in this repository (a test enforces it). The benchmark uses public datasets only;
none of them contains records of individual people.

| Role | Dataset | Source | Licence | SHA-256 of the downloaded archive |
|---|---|---|---|---|
| Classification | Covertype | https://archive.ics.uci.edu/dataset/31/covertype (DOI 10.24432/C50K5N) | CC BY 4.0 | `89a975c2457cd48e824238ae43c5a3cb762e42c4b4078d9b44a4514055105f6d` (11,251,206 bytes) |
| Classification | Online News Popularity | https://archive.ics.uci.edu/dataset/332/online+news+popularity (DOI 10.24432/C5NS3V) | CC BY 4.0 | `dba2ae526f62ccef6f2f8efb53b0268319f4d0d2719bb98d88958cfe31d58a22` (7,476,401 bytes) |
| Time-ordered | Beijing Multi-Site Air Quality | https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data (DOI 10.24432/C5RK5G) | CC BY 4.0 | `b04da438b2f331ac0ffd45aebdfec0d20d2367feb5f6948c4b1f7ce1191e33c4` (8,192,212 bytes) |

## Attribution (required by CC BY 4.0)

- Blackard, J. (1998). Covertype [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C50K5N
- Fernandes, K., Vinagre, P., Cortez, P., Sernadela, P. (2015). Online News Popularity [Dataset].
  UCI Machine Learning Repository. https://doi.org/10.24432/C5NS3V
- Chen, S. (2017). Beijing Multi-Site Air Quality [Dataset]. UCI Machine Learning Repository.
  https://doi.org/10.24432/C5RK5G

## Status

`python tools/download_data.py covertype` (or `news`, `beijing`) fetches the archive once into `data/` and
prints its SHA-256, and fails if it differs from the value above (pinned on 2026-10-10; a mismatching file
is set aside, not used). Each loader independently checks the layout of its file: row counts, columns,
one-hot groups, gap-free hourly series. Tests and the benchmark never touch the network; they run on
synthetic generators and on the local cache.
