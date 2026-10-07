# data

Nothing in here is tracked by git. The directory layout is, so that everyone's
paths match.

```
raw/     exactly what you were given: train.csv, train/, the annotation tables
work/    everything you derive from it: resized copies, feature tables, caches
```

Treat `raw/` as read only. If a script needs a different form of the data it
writes it to `work/`, so that deleting `work/` and regenerating it is always
safe.
