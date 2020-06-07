#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 17:21:26 2019

@author: marcogancitano
"""
from datetime import datetime
import numpy as np
import scrapy
from scrapy.spiders import SitemapSpider

exclude_list = ['/travel/','/specials/','/shows/','/videos/','/style/','/profiles/','/gallery/']

class cnnSpider(SitemapSpider):
    name = "cnn"
    sitemap_urls = ['https://www.cnn.com/sitemaps/cnn/index.xml']

    def sitemap_filter(self, entries):
        for entry in entries:
            date_time = datetime.strptime(entry['lastmod'], '%Y-%m-%d')
            if date_time.year >= 2017:
                yield scrapy.Request(entry.url,callback = self.parse)
    
    
    def parse(self, response):
        try:
            update_year = int(response.css('p[class=update-time] ::text').get().split(",")[2])
        except:
            update_year = 0
        
        if not any([exclude in response.url for exclude in exclude_list]) and update_year >= 2017:
            data = {'url':response.url}
            article = response.css('div[class=l-container]') 
            data['title'] = article.css('h1 ::text').get()
            
            real_section = response.url.split('cnn.com')[1].split('/')[1]
            try:
                real_section = int(real_section)
                data['real_section']= response.url.split('cnn.com')[1].split('/')[4]
            except:
                data['real_section']= real_section

            data['author'] = article.css('span[class=metadata__byline__author]').css('a[href] ::text').get()
            data['edition'] = int ("edition.cnn" in response.url)
            
            selectors = []
            selectors.extend(article.css('p[class="zn-body__paragraph speakable"] ::text'))
            selectors.extend(article.css('div[class="zn-body__paragraph speakable"] ::text'))
            selectors.extend(article.css('div[class="zn-body__paragraph"] ::text'))

            non_unique_text = [selector.get().strip() for selector in selectors]
            
            indexes = np.unique(non_unique_text, return_index=True)[1]
            unique_text = [non_unique_text[index] for index in sorted(indexes)]
            
            data['article'] = " ".join(unique_text)
            
            yield data
        