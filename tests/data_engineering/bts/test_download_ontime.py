import io
import zipfile
from datetime import date

import pytest

from data_engineering.bts import download_ontime as downloader


FORM = '''<form id="form1">
<input type="hidden" name="__VIEWSTATE" value="state">
<select name="cboYear"><option value="2025">2025</option></select>
<select name="cboPeriod"><option value="2">February</option></select>
<input type="checkbox" name="chkDownloadZip">
''' + ''.join(
    f'<input type="checkbox" name="{name}">'
    for name in ("YEAR", "MONTH", "FL_DATE", "ARR_DELAY", "EXTRA_FIELD")
) + '</form>'
CSV = b'YEAR,MONTH,FL_DATE,ARR_DELAY,EXTRA_FIELD\n2025,2,2/1/2025 12:00:00 AM,-5,test\n'
MONTH = date(2025, 2, 1)


class Response:
    def __init__(self, content=b'', text=''):
        self.content, self.text = content, text

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def raise_for_status(self):
        pass

    def iter_content(self, **kwargs):
        yield self.content


class Session:
    def __init__(self, csv=CSV, payload=None):
        archive = io.BytesIO()
        with zipfile.ZipFile(archive, 'w') as zipped:
            zipped.writestr('T_ONTIME_REPORTING.csv', csv)
        self.payload = archive.getvalue() if payload is None else payload
        self.posts = []

    def get(self, *args, **kwargs):
        return Response(text=FORM)

    def post(self, *args, **kwargs):
        self.posts.append(kwargs['data'])
        return Response(content=self.payload)


@pytest.mark.parametrize(('start', 'end', 'expected'), [
    ('2025-01-01', '2025-03-01', ['2025-01-01', '2025-02-01']),
    ('2025-01-31', '2025-02-02', ['2025-01-01', '2025-02-01']),
    ('2025-12-15', '2026-01-01', ['2025-12-01']),
    ('2024-02-29', '2024-03-01', ['2024-02-01']),
])
def test_month_boundaries(start, end, expected):
    assert list(downloader.months_in_range(date.fromisoformat(start), date.fromisoformat(end))) == [
        date.fromisoformat(value) for value in expected
    ]


def test_cli_rejects_empty_interval():
    with pytest.raises(SystemExit):
        downloader.parse_args(['2025-02-01', '2025-02-01'])


def test_cli_defaults_to_v2_root():
    args = downloader.parse_args(['2025-02-01', '2025-03-01'])
    assert args.output_dir == downloader.OUTPUT_DIR / 'data/raw/bts/v2'


def test_download_all_fields_and_skip_complete_file(tmp_path):
    session = Session()
    path = downloader.download_month(MONTH, output_dir=tmp_path, session=session)
    assert path == tmp_path / 'year=2025/month=02/T_ONTIME_REPORTING.csv'
    assert path.read_bytes() == CSV
    assert session.posts[0]['EXTRA_FIELD'] == 'on'
    assert session.posts[0]['__VIEWSTATE'] == 'state'
    assert 'chkDownloadZip' not in session.posts[0]
    downloader.download_month(MONTH, output_dir=tmp_path, session=session)
    assert len(session.posts) == 1
    downloader.download_month(MONTH, output_dir=tmp_path, session=session, overwrite=True)
    assert len(session.posts) == 2


def test_refreshes_old_subset(tmp_path):
    path = tmp_path / 'year=2025/month=02/T_ONTIME_REPORTING.csv'
    path.parent.mkdir(parents=True)
    path.write_text('YEAR,MONTH\n2025,2\n')
    downloader.download_month(MONTH, output_dir=tmp_path, session=Session())
    assert path.read_bytes() == CSV


@pytest.mark.parametrize('session', [
    Session(payload=b'<html>Unavailable</html>'),
    Session(csv=CSV.replace(b'2025,2,', b'2025,3,')),
    Session(csv=b'YEAR,MONTH\n2025,2\n'),
    Session(csv=CSV.splitlines(keepends=True)[0]),
])
def test_failed_download_preserves_existing_file(tmp_path, session):
    path = tmp_path / 'year=2025/month=02/T_ONTIME_REPORTING.csv'
    path.parent.mkdir(parents=True)
    path.write_bytes(b'original')
    with pytest.raises(ValueError):
        downloader.download_month(MONTH, output_dir=tmp_path, session=session, overwrite=True)
    assert path.read_bytes() == b'original'
    assert list(path.parent.iterdir()) == [path]
