---
title: ml-data-guard
emoji: 🛡️
colorFrom: red
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Inject a data fault and see which validation check finds it
---

# ml-data-guard

Web service of [ml-data-guard](https://github.com/slastrzelec/ml-data-guard): break a synthetic
batch and see which data-validation rule notices, or check a CSV file against an inferred schema.
Uploads are processed in memory and not stored. API documentation: `/docs`.
