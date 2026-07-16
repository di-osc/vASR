#!/usr/bin/env python3
"""Build an asr-data AudioDB from the real-estate call workbook.

The workbook's full-call URL is stored as the Audio source. Dialogue is
cleaned and routed to three channel timelines:

* left: AI and human customer-service utterances
* right: user utterances
* mono: every utterance in conversation order

The workbook has no utterance timestamps, so each non-empty timeline contains
one zero-length transcription annotation instead of invented timing data.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZipFile

from asr_data import Audio, AudioDB


MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
SPEAKER_PREFIX = re.compile(r"^\s*(?:\[([^]]+)\]|【([^】]+)】)\s*")
USER_SPEAKERS = {"用户"}
SERVICE_SPEAKERS = {"AI", "人工客服"}
ANNOTATION_SOURCE = "xlsx_dialogue"


def read_xlsx(path: Path) -> list[dict[str, str]]:
    """Read the first worksheet using only the Python standard library."""
    with ZipFile(path) as archive:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            shared = [
                "".join(node.text or "" for node in item.iter(f"{{{MAIN_NS}}}t"))
                for item in root.findall(f"{{{MAIN_NS}}}si")
            ]

        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        relationships = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        targets = {
            item.attrib["Id"]: item.attrib["Target"] for item in relationships
        }
        sheet = next(iter(workbook.find(f"{{{MAIN_NS}}}sheets")))
        target = targets[sheet.attrib[f"{{{REL_NS}}}id"]]
        if target.startswith("/"):
            target = target.lstrip("/")
        elif not target.startswith("xl/"):
            target = f"xl/{target}"

        sheet_root = ET.fromstring(archive.read(target))
        parsed_rows: list[dict[str, str]] = []
        for row in sheet_root.find(f"{{{MAIN_NS}}}sheetData"):
            values: dict[str, str] = {}
            for cell in row.findall(f"{{{MAIN_NS}}}c"):
                column_match = re.match(r"[A-Z]+", cell.attrib["r"])
                if column_match is None:
                    continue
                column = column_match.group(0)
                cell_type = cell.attrib.get("t")
                inline = cell.find(f"{{{MAIN_NS}}}is")
                value_node = cell.find(f"{{{MAIN_NS}}}v")
                if inline is not None:
                    value = "".join(
                        node.text or "" for node in inline.iter(f"{{{MAIN_NS}}}t")
                    )
                elif value_node is None:
                    value = ""
                elif cell_type == "s":
                    value = shared[int(value_node.text)]
                else:
                    value = value_node.text or ""
                values[column] = value
            parsed_rows.append(values)

    headers = parsed_rows[0]
    return [
        {headers.get(column, column): value for column, value in row.items()}
        for row in parsed_rows[1:]
    ]


def parse_dialogue(raw: str) -> list[tuple[str, str]]:
    """Return ``(speaker, text)`` entries with the speaker marker removed."""
    if not raw or raw.strip() in {"", "null"}:
        return []
    messages = json.loads(raw)
    if not isinstance(messages, list):
        raise ValueError("dialogue must be a JSON list")

    utterances: list[tuple[str, str]] = []
    for message in messages:
        if not isinstance(message, str):
            continue
        match = SPEAKER_PREFIX.match(message)
        if match is None:
            raise ValueError(f"dialogue message has no speaker marker: {message!r}")
        speaker = match.group(1) or match.group(2)
        text = SPEAKER_PREFIX.sub("", message, count=1).strip()
        if text:
            utterances.append((speaker, text))
    return utterances


def dialogue_timelines(record: dict[str, str]) -> dict[str, str]:
    dialogue = parse_dialogue(record.get("对话内容", ""))
    dialogue.extend(parse_dialogue(record.get("人工对话内容", "")))

    unknown = sorted(
        {speaker for speaker, _ in dialogue}
        - USER_SPEAKERS
        - SERVICE_SPEAKERS
    )
    if unknown:
        raise ValueError(f"unknown dialogue speakers: {unknown}")

    return {
        "left": "".join(
            text for speaker, text in dialogue if speaker in SERVICE_SPEAKERS
        ),
        "right": "".join(
            text for speaker, text in dialogue if speaker in USER_SPEAKERS
        ),
        "mono": "".join(text for _, text in dialogue),
    }


def add_transcript(audio: Audio, channel: str, text: str) -> None:
    timeline = audio.ensure_timeline(channel)
    if text:
        timeline.add_transcription(
            0,
            0,
            text,
            source=ANNOTATION_SOURCE,
            source_kind="stage",
            language="zh",
        )


def build(workbook: Path, output: Path, limit: int | None = None) -> None:
    records = read_xlsx(workbook)
    if limit is not None:
        records = records[:limit]

    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")

    db = AudioDB(str(output))
    db.set_metadata("source_workbook", workbook.name)
    db.set_metadata("record_count_requested", len(records))
    db.set_metadata("annotation_source", ANNOTATION_SOURCE)
    db.set_metadata(
        "channel_mapping",
        {
            "left": "customer_service",
            "right": "user",
            "mono": "full_dialogue",
        },
    )

    inserted = 0
    failures: list[dict[str, str]] = []
    for index, record in enumerate(records, 1):
        call_id = record.get("callid", "").strip()
        url = (
            record.get("全程录音链接", "").strip()
            or record.get("全程录音", "").strip()
        )
        try:
            if not call_id or not url or url == "null":
                raise ValueError("missing callid or full-call recording URL")

            texts = dialogue_timelines(record)
            audio = Audio.from_url(url, id=call_id)
            for channel in ("left", "right", "mono"):
                add_transcript(audio, channel, texts[channel])

            audio.metadata["call_time"] = record.get("时间", "")
            audio.metadata["business_line"] = record.get("业务线", "")
            audio.metadata["hotline"] = record.get("热线号码", "")
            audio.metadata["caller"] = record.get("来电号码", "")
            audio.metadata["session_status"] = record.get("会话状态", "")
            audio.metadata["user_reply_count"] = record.get("用户回复轮次", "")
            audio.validate()
            db.insert(audio)
            inserted += 1
            print(f"[{index}/{len(records)}] OK {call_id}")
        except Exception as error:
            failures.append({"callid": call_id, "error": str(error), "url": url})
            print(f"[{index}/{len(records)}] FAILED {call_id}: {error}")

    db.set_metadata("record_count_inserted", inserted)
    db.set_metadata("failures", failures)
    print(f"completed: inserted={inserted} failed={len(failures)} output={output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("workbook", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    build(args.workbook, args.output, args.limit)


if __name__ == "__main__":
    main()
