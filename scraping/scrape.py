from selenium import webdriver
from bs4 import BeautifulSoup
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import csv
import requests

def click_through_start(driver):
    button = driver.find_element(By.ID, "details-button")
    button.click()
    driver.implicitly_wait(0.5)
    link = driver.find_element(By.ID, "proceed-link")
    driver.implicitly_wait(0.5)
    link.click()
    driver.implicitly_wait(0.5)
def find_on_page(page):
    lines = page.find_elements(By.CLASS_NAME, "glossedText")
    for line in lines:
        translation = line.find_element(By.TAG_NAME, 'p')
        ramsesLine = line.find_element(By.CLASS_NAME, 'ramsesLine')
        transl = ramsesLine.find_element(By.CLASS_NAME, 'transliteration')
        trans = ramsesLine.find_element(By.CLASS_NAME, 'translation')
        transl_words = transl.find_elements(By.CLASS_NAME, "word")
        trans_words = trans.find_elements(By.CLASS_NAME, "word")
        for word in transl_words:
            print(word.text, end=' _ ')
        print()
        for word in trans_words:
            print(word.text, end=' _ ')
        print()
        print(translation.text)
        print()

options = Options()
options.add_argument('--headless=new')
driver = webdriver.Chrome(options=options)

with open('data.csv', mode='w', newline='',  encoding="utf-8") as data_csv:
    scraped_data = csv.writer(data_csv, delimiter=',')
    scraped_data.writerow(["transliteration", "translation", "price", "pictures"])
    url = 'https://ramses.ulg.ac.be/text'
    driver.get(url)
    driver.implicitly_wait(0.5)

    click_through_start(driver)

    # create a file with urls of every soubor

    num_texts = 0
    run = True
    for i in range(3):
        texts = driver.find_elements(By.CLASS_NAME, 'clickable')
        # for text in texts:
        #     text.get_attribute('href')
        num_texts += len(texts)
        print(num_texts)
        # for i in range(len(texts)):
        #     driver.get(url)
        #     texts = driver.find_elements(By.CLASS_NAME, 'clickable')
        #     texts[i].click()
        #     driver.implicitly_wait(0.5)
        #     find_on_page(driver)
        next = driver.find_element(By.CLASS_NAME, 'next')
        next = next.find_element(By.TAG_NAME, 'a')
        if next:
            next.click()
            driver.implicitly_wait(0.5)
        else:
            run = False
    print("num texts:", num_texts)

    # print("tags")
    # tags = driver.find_elements(By.TAG_NAME, "tr")
    # for tag in tags:
    #     print(tag.text)

    #
    # for reality in realities:
    #     title = reality.find('span', class_='name ng-binding').text
    #     name = 'Byt'
    #     size = title.split("u ")[1]
    #     rooms = size.split(" ")[0]
    #     size = size.split(" ")[1].split(" ")[0]
    #     if '(' in title:
    #         note = title.split('(')[1].split(')')[0]
    #         size = size.split(" (")[0]
    #     else:
    #         note = ''
    #     price = reality.find('span', class_='norm-price ng-binding').text
    #     price = price.split(" Kč")[0]
    #     try:
    #         price = price.split(" ")[0] + price.split(" ")[1]
    #     except ValueError or IndexError:
    #         price = '0'
    #     locality = reality.find('span', class_='locality ng-binding').text
    #     url = 'https://www.sreality.cz' + reality.find('a', class_='title')['href']
    #     imgurl = reality.find('img')['src']
    #     print(f'{name}, {rooms}, {size}, {price}, {locality}, {note}, {url} ({imgurl})')
    #     reality_writer.writerow([size, rooms, price, locality, note, url, imgurl])
    #     # maybe add some numbering system?

    #driver.find_element(By.CLASS_NAME, "").click()