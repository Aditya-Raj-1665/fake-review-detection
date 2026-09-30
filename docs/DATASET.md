# Dataset provenance and limitations

The real baseline uses the **Cornell Deceptive Opinion Spam Corpus v1.4**, imported
from [Rachael Tatman's Kaggle mirror](https://www.kaggle.com/datasets/rtatman/deceptive-opinion-spam-corpus).
The importer has been exercised against the 1,600-row download. Its archive SHA-256 is:

```text
f93674ef50e365886aeef5e651ac8990e8430d36e439ae1ca5fa6962a76a0a8e
```

The research is described by [Ott et al., 2011](https://aclanthology.org/P11-1032/)
and [Ott et al., 2013](https://aclanthology.org/N13-1053/). The first paper covers positive
deceptive hotel reviews; the second extends the work to negative reviews. The deceptive
examples were commissioned from crowd workers; truthful examples were collected from
online review sources. These labels do not prove the authenticity of every scraped review.

The downloaded CSV contains text, deception label, hotel, polarity, and source. It
represents 20 Chicago hotels, positive and negative sentiment, and 800 rows per label
before deduplication. Only text and the normalized label are model inputs. The importer
preserves other columns for future group-aware analyses, but v1's trainer ignores them.
It maps `truthful` to `genuine` and `deceptive` to `deceptive` explicitly.

The author-hosted download failed with HTTP 403 during implementation. An ACL attachment
also did not return a readable archive in this environment. The working Kaggle mirror
is therefore documented explicitly instead of pretending to be an author-hosted endpoint.
A checksum mismatch causes import to stop rather than accept a silent upstream change.
No archive paths are extracted and no raw corpus is committed to the repository.

Use this corpus for appropriately attributed learning/research. Upstream dataset and
review-content rights remain with their owners; this project's code does not grant a
license to redistribute the raw reviews. Review source terms before redistribution or
commercial reuse. The importer writes a provenance sidecar beside the local CSV.

## What the benchmark does and does not measure

The included split removes exact/normalized duplicate text, but does not group by hotel
or author. Names, collection source, writing instructions, and historical style can
provide shortcuts. Crowdsourced fabrications differ from naturally occurring coordinated
spam and modern AI-written reviews. Product reviews, other languages, different deception
prevalence, and future writing styles may behave very differently.

The random held-out score measures classification of this corpus under this split. It
is not an estimate of accuracy over all reviews on the internet. Use independent domains,
grouped holdouts, human error analysis, and uncertainty intervals before stronger claims.

`data/demo/reviews.csv` is deliberately invented, simplistic plumbing-test data. Its
sentiment/exaggeration patterns are not valid fraud rules; never report its metrics as
evidence of model quality.
