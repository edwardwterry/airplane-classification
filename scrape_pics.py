#!usr/bin/env python

import urllib2
import requests
import os
hdr = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11',
       'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
       'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3',
       'Accept-Encoding': 'none',
       'Accept-Language': 'en-US,en;q=0.8',
       'Connection': 'keep-alive'}

image_addresses = []

airplanes = ['737','777','787']
start_page = 20
end_page = 60

for a in airplanes:
    index = 84 * start_page
    for p in range(start_page, end_page):
        site = "https://www.airliners.net/search?keywords=" + a + \
                       "&sortBy=dateAccepted&sortOrder=desc&perPage=84&display=detail&page=" + str(p + 1)
        print site
        req = urllib2.Request(site, headers=hdr)
        try:
            page = urllib2.urlopen(req)
        except urllib2.HTTPError, e:
            print e.fp.read()

        content = page.read().splitlines()
        for line in content:
            if line.find("lazy-load") > 0:
                # print line
                start = line.find('https')
                end = line.find('.jpg') + len('.jpg')
                # print start, end
                address = line[start: end]
                print "Address", address
                image_addresses.append(address)
                print "Saving to", a + "_" + str(index) + ".jpg"
                # https://stackoverflow.com/questions/30229231/python-save-image-from-url
                img_data = requests.get(address).content
                with open("images/" + a + "_" + str(index) + ".jpg", 'wb') as handler:
                    handler.write(img_data)
                index += 1