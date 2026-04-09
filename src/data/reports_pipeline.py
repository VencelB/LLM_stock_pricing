import re
import requests
import json
import zipfile

from io import BytesIO
from pypdf import PdfReader
from multiprocessing import Pool


markers = "|".join(
    [
        "ebitda",
        "price",
        "earning",
        "roe",
        "p/e",
        "p/bv",
        "share",
        "earn",
        "asset",
        "liabilt",
        "net",
        "debt",
        "commod",
        "secur" "выручк",
        "прибыл",
        "чист",
        "актив",
        "пассив",
        "запас",
        "дебитор",
        "кредитор",
        "займ",
        "налог",
        "ден",
        "амортизац",
        "истощ",
        "доход",
        "расход",
        "процент",
        "оборот",
        "инвест",
        "поступления",
        "убыток",
        "возмещ",
        "депозит",
        "ликвид",
        "обязатеств",
        "операционн",
        "акци",
        "капитал",
        "резерв",
        "курс",
        "дол",
        "краткосрочн",
        "выплат",
        "аренд",
        "офис",
        "долгосрочн",
    ]
)

numeric = "|".join(["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"])

markers = re.compile(markers)
numeric = re.compile(numeric)


def is_pdf_by_magic_bytes(binary_data):
    search_limit = min(len(binary_data), 1024)
    for i in range(search_limit - 3):
        if binary_data[i : i + 4] == b"%PDF":
            return True
    return False


def extract_zip(binary_data):
    with BytesIO(binary_data) as b:
        with zipfile.ZipFile(b, "r") as z:
            file_to_read = z.namelist()[0]
            with z.open(file_to_read) as f:
                content = f.read()
    return content


def download(reports_metadata):
    b = re.search("Дата отчета", reports_metadata.text).start()
    e = re.search("Валюта отчета", reports_metadata.text).start()
    dates = re.findall(r"\d{2}\.\d{2}\.\d{4}", reports_metadata.text[b:e])

    url_pattern = (
        r'<a href=".+" target="_blank" title="Открыть файл отчета" class="icon pdf">'
    )
    urls = [p[9:-63] for p in re.findall(url_pattern, reports_metadata.text)]
    res = []
    for d, url in zip(dates, urls):
        try:
            r = requests.get(url)
            if r.status_code == 200:
                year = d.split(".")[-1]
                if is_pdf_by_magic_bytes(r.content):
                    res.append((year, r.content))
                else:
                    content = extract_zip(r.content)
                    if is_pdf_by_magic_bytes(content):
                        res.append((year, content))
        except:
            continue
    return res


def text_pdf(binary_pdf):
    reader = PdfReader(BytesIO(binary_pdf))

    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text() + "\n"
    return full_text


def extract_snippents(text, min_rows=5, max_rows=16, top_k=5, min_numeric=15):
    snippets = []
    for s in text.split("\n \n"):
        tmp = s.split("\n")
        if len(tmp) >= min_rows:
            snippets += [
                "\n".join(tmp[i : i + max_rows])
                for i in range(0, len(tmp), max_rows)
                if (len(tmp) >= (i + min_rows))
            ]

    return sorted(
        [
            (len(re.findall(markers, s.lower())), re.sub(r"[ \t]+", " ", s))
            for s in snippets
            if (
                len(re.findall(numeric, s)) >= min_numeric
                and len(re.findall(markers, s.lower())) > 0
            )
        ],
        reverse=True,
    )[:top_k]


def process_tasks(tasks):
    for stock, task in tasks:
        try:
            raw_reports = download(task)
            for year, raw_report in raw_reports:
                report = text_pdf(raw_report)
                report_snippets = extract_snippents(report)
                with open(
                    f"../../data/{stock}_{year}.jsonl", "w", encoding="utf-8"
                ) as out:
                    for s in report_snippets:
                        out.write(
                            json.dumps({"name": stock, "date": year, "content": s[1]})
                            + "\n"
                        )
        except:
            pass


def main():
    stock_list = requests.get("https://smart-lab.ru/q/shares_fundamental5/")

    stock_report_pattern = r"/q/\w+/f/y/"
    stock_report_pages = [
        (stock.split("/")[2], f"https://smart-lab.ru/{stock}MSFO/")
        for stock in re.findall(stock_report_pattern, stock_list.text)
    ]
    stock_report_pages = [
        (stock, requests.get(stock_url)) for stock, stock_url in stock_report_pages
    ]

    n_proc = 6
    task_size = (len(stock_report_pages) + n_proc - 1) // n_proc
    tasks_pool = [
        stock_report_pages[i : i + task_size]
        for i in range(0, len(stock_report_pages), task_size)
    ]

    with Pool(processes=n_proc) as pool:
        pool.map(process_tasks, tasks_pool[:6])


if __name__ == "__main__":
    main()
