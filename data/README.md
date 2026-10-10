# Data

No data file is tracked in this repository (a test enforces it). The benchmark uses public datasets only;
none of them contains records of individual people.

| Role | Dataset | Source | Licence | SHA-256 of the downloaded archive |
|---|---|---|---|---|
| Classification | Covertype | https://archive.ics.uci.edu/dataset/31/covertype (DOI 10.24432/C50K5N) | CC BY 4.0 | `89a975c2457cd48e824238ae43c5a3cb762e42c4b4078d9b44a4514055105f6d` (11,251,206 bytes) |
| Regression | Online News Popularity | https://archive.ics.uci.edu/dataset/332/online+news+popularity (DOI 10.24432/C5NS3V) | CC BY 4.0 | phase 3 |
| Time-ordered | Beijing Multi-Site Air Quality | https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data (DOI 10.24432/C5RK5G) | CC BY 4.0 | phase 3 |

## Attribution (required by CC BY 4.0)

- Blackard, J. (1998). Covertype [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C50K5N
- Fernandes, K., Vinagre, P., Cortez, P., Sernadela, P. (2015). Online News Popularity [Dataset].
  UCI Machine Learning Repository. https://doi.org/10.24432/C5NS3V
- Chen, S. (2017). Beijing Multi-Site Air Quality [Dataset]. UCI Machine Learning Repository.
  https://doi.org/10.24432/C5RK5G

## Status

`python tools/download_data.py covertype` fetches the archive once into `data/covertype/` and prints its
SHA-256, and fails if it differs from the value above (pinned on 2026-10-10; the mismatching file is set
aside, not used). The loader independently checks the documented shape (581,012 rows, 55
columns) and the one-hot structure. Tests and the benchmark never touch the network; they run on
synthetic generators and on the local cache.
