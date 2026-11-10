import argparse
import html
import logging
import re
import time
from collections import Counter

import apache_beam as beam
from apache_beam.io import fileio
from apache_beam.io.filesystems import FileSystems
from apache_beam.options.pipeline_options import PipelineOptions

HREF_RE = re.compile(r'<a\s+[^>]*href\s*=\s*["\']?([^"\'\s>]+)', re.IGNORECASE)
TAG_RE = re.compile(r"<[^>]+>")
WORD_RE = re.compile(r"[a-z0-9']+")


def basename(path):
    return path.rstrip("/").split("/")[-1].split("?")[0].split("#")[0]


def read_file(readable_file):
    name = basename(readable_file.metadata.path)
    text = readable_file.read().decode("utf-8", errors="ignore")
    return name, text


def outgoing_links(name_text):
    name, text = name_text
    return [basename(h) for h in HREF_RE.findall(text)]


def trigram_counts(name_text):
    _, text = name_text
    words = WORD_RE.findall(html.unescape(TAG_RE.sub(" ", text)).lower())
    counts = Counter(" ".join(words[i:i + 3]) for i in range(len(words) - 2))
    return counts.items()


def rank_key(kv):
    name, count = kv
    return (count, tuple(-ord(c) for c in name) + (1,))


def format_top(top_list, unit):
    return [f"{i}. {name}: {count} {unit}" for i, (name, count) in enumerate(top_list, 1)]


def top5_to_file(pcoll, label, unit, out_path):
    return (
        pcoll
        | f"Top5 {label}" >> beam.combiners.Top.Of(5, key=rank_key)
        | f"Format {label}" >> beam.FlatMap(format_top, unit)
        | f"Write {label}" >> beam.io.WriteToText(out_path, shard_name_template="")
    )


def run(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="gs://cs528-pagerank-eddielu-1790373540/files/*.html")
    parser.add_argument("--output", required=True)
    args, beam_args = parser.parse_known_args(argv)
    options = PipelineOptions(beam_args)
    runner = options.get_all_options().get("runner") or "DirectRunner"
    out = args.output.rstrip("/")

    start = time.time()
    with beam.Pipeline(options=options) as p:
        files = (
            p
            | "Match" >> fileio.MatchFiles(args.input)
            | "ReadMatches" >> fileio.ReadMatches()
            | "ToNameText" >> beam.Map(read_file)
        )

        incoming = (
            files
            | "Targets" >> beam.FlatMap(outgoing_links)
            | "CountIncoming" >> beam.combiners.Count.PerElement()
        )
        outgoing = files | "CountOutgoing" >> beam.Map(
            lambda nt: (nt[0], len(outgoing_links(nt))))
        trigrams = (
            files
            | "Trigrams" >> beam.FlatMap(trigram_counts)
            | "SumTrigrams" >> beam.CombinePerKey(sum)
        )

        top5_to_file(incoming, "incoming", "incoming links", f"{out}/incoming_links.txt")
        top5_to_file(outgoing, "outgoing", "outgoing links", f"{out}/outgoing_links.txt")
        top5_to_file(trigrams, "trigrams", "times", f"{out}/trigrams.txt")

    elapsed = time.time() - start

    for title, fname in [("Top 5 files by incoming links", "incoming_links.txt"),
                         ("Top 5 files by outgoing links", "outgoing_links.txt"),
                         ("Top 5 word trigrams", "trigrams.txt")]:
        with FileSystems.open(f"{out}/{fname}") as f:
            print(f"\n{title}\n{f.read().decode().strip()}")
    print(f"\nRunner: {runner}\nInput: {args.input}\nTotal time: {elapsed:.1f} seconds")


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.WARNING)
    run()
