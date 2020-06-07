# -*- coding: utf-8 -*-

from datetime import date,datetime,timedelta
import itertools

import numpy as np
import scrapy

sections = ["politics","entertainment","the-media","economy",
           "europe","border","middle-east","africa","asia",
           "latin-america","tech","sports",
           "social-justice"]

now = datetime.now()

def perdelta(start, end, delta):
    dates = [start]
    curr = start
    while curr < end:
        curr += delta
        dates.append(curr)
    return dates

dates_to_crawl = perdelta(date(2016, 1, 1), date(now.year, now.month, now.day), timedelta(days=1))

link_tuples = list(itertools.product(dates_to_crawl, sections))
create_links = lambda x: "https://www.breitbart.com/" + x[1] + "/" + str(x[0]).replace('-','/') + "/"

class breitbartSpider(scrapy.Spider):
    name = "breitbart_links"
    start_urls = list(map(create_links,link_tuples))

    def parse(self, response):
        for article in response.css('article'):
            url = "https://www.breitbart.com" + article.css('a ::attr(href)').extract_first()
            yield scrapy.Request(url,callback = self.parse_content)
    
    def parse_content(self,response):
        data = {'url':response.url}
        
        title = np.nan
        while type(title) == type(5.0):
            try:
                article = response.css('article')
                title = article.css('h1 ::text').get()
            except:
                yield scrapy.Request(response.url,callback = self.parse_content)

        data['title'] = title

        real_section = response.url.split('/')[3]
        if real_section == 'clips':
            return
        
        data['real_section'] = real_section

        try:
            data['snippet'] = article.css('h2 ::text').get()
        except:
            data['snippet'] = None
            
        try:
            data['shares'] = int(article.css('span[class=acz5] ::text').get().replace(",",""))
        except:
            data['shares'] = None
        
        try:
            data['author'] = article.css('address ::text').get()
        except:
            data['author'] = None
        
        try:
            datetime_value = datetime.strptime(article.css('time ::text').get(),"%d %b %Y")
            data['published_date'] = str(date(datetime_value.year,datetime_value.month,datetime_value.day))
        except:
            data['published_date'] = None
            
        try:
            p_lst = article.css('p')
            read_flag = 1
            article_text = ""
            tweet_count = 0
            for p in p_lst:
                if p.css('p[class]') != []:
                    if ("rmoreabt" in p.css('p[class]').get()):
                        break
                if p.css('a') != []:
                    if p.css('a[rel]') != []:
                        if ("external" in p.css('a[rel]').get() and "noopener" in  p.css('a[rel]').get() and "@" in p.css('a[rel] ::text').get() and "follow" in p.css('a[rel] ::text').get()):
                            break
                    if p.css('a[class]') != []:
                        if 'twitter-follow-button' in p.css('a[class]').get():
                            break
                if p.css('dir') != []:
                    read_flag = 0
                    tweet_count += 1
                if (read_flag):
                    sentence_lst = p.css('::text').extract()
                    article_text = article_text.strip() + " " + "".join(sentence_lst).strip()
                if p.css('a[href]') != [] and "https://twitter.com/" in p.css('a[href]').get():
                    read_flag = 1
            article_text = article_text.encode('ascii', errors='ignore').decode("utf-8")
        except:
            article_text = None
        data['article_text'] = article_text
        
        yield data
