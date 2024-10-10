import logging
from typing import NamedTuple

from DrissionPage import ChromiumPage
from rich.logging import RichHandler

# 设置日志记录的基本配置
logging.basicConfig(
    # 设置日志级别为INFO
    level="INFO",
    # 设置日志格式为只显示消息
    format="%(message)s",
    # 设置日期格式为只显示时间
    datefmt="[%X]",
    # 设置日志处理器为RichHandler，并设置显示tracebacks和本地变量
    handlers=[RichHandler(rich_tracebacks=True, tracebacks_show_locals=True)],
)
# 获取名为rich的日志记录器
log = logging.getLogger("rich")


class ChapterInfo(NamedTuple):
    # 定义name属性，类型为str
    name: str
    # 定义url属性，类型为str
    url: str


class Index(NamedTuple):
    # 定义name属性，类型为str
    name: str
    # 定义chpts属性，类型为list[ChapterInfo]
    chpts: list[ChapterInfo]


class Crawler:
    # 初始化Crawler类，创建一个ChromiumPage对象
    def __init__(self) -> None:
        self.page = ChromiumPage()

    # 根据url获取Index对象
    def get_index(self, url: str) -> Index:
        # 获取url对应的页面
        self.page.get(url)
        # 获取页面中id为bookName的元素的文本内容，作为书名
        bookname = self.page.ele("#bookName").text
        # 获取页面中class为chapter-name的元素列表
        elems = self.page.s_eles(".chapter-name")
        # 创建一个空列表，用于存储ChapterInfo对象
        urls: list[ChapterInfo] = []
        # 遍历元素列表
        for elem in elems:
            # 获取元素的文本内容，作为章节名
            name = elem.text
            # 获取元素的href属性，作为章节链接
            href = elem.attr("href")
            # 如果href属性不为空，则将ChapterInfo对象添加到urls列表中
            if href is not None:
                urls.append(ChapterInfo(name, href))
        # 返回Index对象，包含书名和章节信息
        return Index(bookname, urls)

    def get_chpt(self, chpt: str) -> str:
        # 获取指定章节的内容
        self.page.get(chpt)
        content: list[str] = []
        # 获取章节标题
        title = self.page.ele(".:title").text
        content.append(title)
        # 获取章节内容
        for elem in self.page.eles(".content-text"):
            content.append(elem.text)
        # 将内容以换行符连接起来
        return "\n".join(content)