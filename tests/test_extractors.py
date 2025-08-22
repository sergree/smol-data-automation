from types import SimpleNamespace

import httpx
import pandas as pd
import pytest

from src.extractors import CSVFile, CSVUrl, JSONFile, RestAPI, get_extractor


class AppCfgStub:
    def get_postresql_connection(self, name): ...


@pytest.mark.asyncio
async def test_get_extractor_mapping():
    e = get_extractor(SimpleNamespace(type="csv_file", path="x.csv"), AppCfgStub())
    assert isinstance(e, CSVFile)


@pytest.mark.asyncio
async def test_csv_file(tmp_path):
    p = tmp_path / "f.csv"
    pd.DataFrame({"a": [1, 2]}).to_csv(p, index=False)
    extr = CSVFile(SimpleNamespace(path=str(p)), AppCfgStub())
    df = await extr.extract()
    assert df.equals(pd.DataFrame({"a": [1, 2]}))


@pytest.mark.asyncio
async def test_json_file(tmp_path):
    p = tmp_path / "d.json"
    p.write_text('[{"a":1},{"a":2}]', encoding="utf-8")
    extr = JSONFile(SimpleNamespace(path=str(p)), AppCfgStub())
    df = await extr.extract()
    assert list(df["a"]) == [1, 2]


@pytest.mark.asyncio
async def test_csv_url(monkeypatch):
    class R:
        def __init__(self, text):
            self.text = text

    class C:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args): ...
        async def get(self, url):
            return R("a\n1\n2\n")

    monkeypatch.setattr(httpx, "AsyncClient", C)
    extr = CSVUrl(SimpleNamespace(url="http://t"), AppCfgStub())
    df = await extr.extract()
    assert list(df["a"]) == [1, 2]


@pytest.mark.asyncio
async def test_rest_api(monkeypatch):
    class R:
        def json(self):
            return [{"a": 1}, {"a": 2}]

    class C:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args): ...
        async def get(self, url):
            return R()

    monkeypatch.setattr(httpx, "AsyncClient", C)
    extr = RestAPI(SimpleNamespace(url="http://t"), AppCfgStub())
    df = await extr.extract()
    assert list(df["a"]) == [1, 2]
