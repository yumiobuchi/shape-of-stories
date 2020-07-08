from lxml import html
import requests
page = requests.get('http://aleph.gutenberg.org/')
tree = html.fromstring(page.content)
trythis = tree.xpath('//td[@title="0"]/text()')
print(trythis)
# prices = tree.xpath('//span[@class="item-price"]/text()')
# print(buyers)
# print(prices)

#tree now contains the whole HTML file in a nice tree structure
#which we can go over two different ways: XPath and CSSSelect.

