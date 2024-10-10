import logging
from typing import NamedTuple

from DrissionPage import ChromiumPage
from rich.logging import RichHandler

# ������־��¼�Ļ�������
logging.basicConfig(
    # ������־����ΪINFO
    level="INFO",
    # ������־��ʽΪֻ��ʾ��Ϣ
    format="%(message)s",
    # �������ڸ�ʽΪֻ��ʾʱ��
    datefmt="[%X]",
    # ������־������ΪRichHandler����������ʾtracebacks�ͱ��ر���
    handlers=[RichHandler(rich_tracebacks=True, tracebacks_show_locals=True)],
)
# ��ȡ��Ϊrich����־��¼��
log = logging.getLogger("rich")


class ChapterInfo(NamedTuple):
    # ����name���ԣ�����Ϊstr
    name: str
    # ����url���ԣ�����Ϊstr
    url: str


class Index(NamedTuple):
    # ����name���ԣ�����Ϊstr
    name: str
    # ����chpts���ԣ�����Ϊlist[ChapterInfo]
    chpts: list[ChapterInfo]


class Crawler:
    # ��ʼ��Crawler�࣬����һ��ChromiumPage����
    def __init__(self) -> None:
        self.page = ChromiumPage()

    # ����url��ȡIndex����
    def get_index(self, url: str) -> Index:
        # ��ȡurl��Ӧ��ҳ��
        self.page.get(url)
        # ��ȡҳ����idΪbookName��Ԫ�ص��ı����ݣ���Ϊ����
        bookname = self.page.ele("#bookName").text
        # ��ȡҳ����classΪchapter-name��Ԫ���б�
        elems = self.page.s_eles(".chapter-name")
        # ����һ�����б����ڴ洢ChapterInfo����
        urls: list[ChapterInfo] = []
        # ����Ԫ���б�
        for elem in elems:
            # ��ȡԪ�ص��ı����ݣ���Ϊ�½���
            name = elem.text
            # ��ȡԪ�ص�href���ԣ���Ϊ�½�����
            href = elem.attr("href")
            # ���href���Բ�Ϊ�գ���ChapterInfo������ӵ�urls�б���
            if href is not None:
                urls.append(ChapterInfo(name, href))
        # ����Index���󣬰����������½���Ϣ
        return Index(bookname, urls)

    def get_chpt(self, chpt: str) -> str:
        # ��ȡָ���½ڵ�����
        self.page.get(chpt)
        content: list[str] = []
        # ��ȡ�½ڱ���
        title = self.page.ele(".:title").text
        content.append(title)
        # ��ȡ�½�����
        for elem in self.page.eles(".content-text"):
            content.append(elem.text)
        # �������Ի��з���������
        return "\n".join(content)