# text-alpha
Python implementation of character-level, textual inter-annotator agreement with Krippendorff's alpha.

The implementation is based on the formulation of Krippendorff's alpha for textual content analysis in:

[Krippendorff, Klaus. "Measuring the reliability of qualitative text analysis data."
Quality and quantity 38 (2004): 787-800.](https://repository.upenn.edu/cgi/viewcontent.cgi?article=1042&context=asc_papers)

The worked example included in the appendix of [Krippendorff (2004)]((https://repository.upenn.edu/cgi/viewcontent.cgi?article=1042&context=asc_papers)) is included in the sample data for testing: [`figure-2-example`](https://github.com/zyocum/text-alpha/tree/main/sample-data/figure-2-example)

The expected data format is character-level standoff annotations in the [Basis Technology Annotated Data Model](https://github.com/basis-technology-corp/annotated-data-model).

## Setup
`alpha.py` requires Python 3.8+.  There are no other dependencies.

If you prefer to reformat the tabular output in a more human-friendly way, you can install [`tabulate`](https://github.com/astanin/python-tabulate) with the provided `requirements.txt`:

```shell
$ python -m venv text-alpha
$ source text-alpha/bin/activate
$ pip3 install -r requirements.txt
...
```

## Example Usage
```
$ ./alpha.py -h
usage: alpha.py [-h] [-g GLOB_PATTERN] [-R] [-l LABELS [LABELS ...]] [-p] [-v] annotators [annotators ...]

This script computes Krippendorff's alpha, a measure of inter-annotator agreement (a.k.a. IAA or inter-annotator reliability). The formulation of IAA here
considers textual annotations at a character level, so the expected input are parallel corpora with textual annotations where labels are assigned to spans of
text marked with character offsets. The formula for expected and observed agreement, and the per-label and overall alpha scores are based on this paper:
Krippendorff, Klaus. "Measuring the reliability of qualitative text analysis data." Quality and quantity 38 (2004): 787-800.
https://repository.upenn.edu/cgi/viewcontent.cgi?article=1042&context=asc_papers

positional arguments:
  annotators            paths to annotator directories (at least two)

optional arguments:
  -h, --help            show this help message and exit
  -g GLOB_PATTERN, --glob-pattern GLOB_PATTERN
                        glob pattern for matching annotation files within each annotator directory (default: *.adm.json)
  -R, --non-recursive   if this is specified, annotator directories will not be searched recursively for annotatin files (default: True)
  -l LABELS [LABELS ...], --labels LABELS [LABELS ...]
                        allow-list of labels to check (by default, all labels in the data will be assessed) (default: None)
  -p, --pair-wise       assess IAA pair-wise for each pair of annotators (default: False)
  -v, --verbose         write verbose output to stderr (default: False)
```

If you run `alpha.py` providing paths to annotator directories containing parallel `*.adm.json` files, then the script with compute and report character-level alpha scores for each labeled entity type in the data, as well as an overall alpha score across all entity types:

```
$ ./alpha.py sample-data/figure-2-example/{i,j} | tabulate -s $'\t' -F 0.3
---------  -----
k          1.0
c          0.729
overall α  0.859
---------  -----
```

If you are interested in a subset of labels, you can use the `-l/--labels` option to restrict the labels of interest.  (Note that due to the way that observed and expected agreement are computed, the overall alpha computed over a subset of labels may differ from the overall alpha over all labels.)

For the example from Figure 2 in the paper, the alpha score for label k is 1.0 since the annotators are in perfect agreement on spans labeled as k:

```
$ ./alpha.py sample-data/figure-2-example/{i,j} -l k | tabulate -s $'\t' -F 0.3  
---------  ---
k          1.0
overall α  1.0
---------  ---
```

The annotators disagreed some on label c:

```
$ ./alpha.py sample-data/figure-2-example/{i,j} -l c | tabulate -s $'\t' -F 0.3 
---------  -----
c          0.729
overall α  0.729
---------  -----
```
