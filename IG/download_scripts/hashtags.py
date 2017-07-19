import re
import urllib.request
from bs4 import BeautifulSoup

#Script to download top 10k hashtags from top-hashtags.com
#Each page contains 100 hashtags. 100 pages are scanned

hashtags = []

for page in range(100):
	print('page ', page)
	url = 'https://top-hashtags.com/instagram/' + str(page) + '01/'
	req = urllib.request.Request(url, headers={'User-Agent' : "Magic Browser"})
	html = urllib.request.urlopen(req).read()
	soup = BeautifulSoup(html, 'html.parser')
	anchors = soup.findAll('a', href=re.compile('/hashtag/'))
	for anchor in anchors:
		a = anchor['href']
		a = a[:-1] #remove last '/'
		a = a[a.rfind('/')+1:]
		hashtags.append(a)
		
