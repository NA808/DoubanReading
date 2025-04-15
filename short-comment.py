import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
import time
import random
import json


def create_directory(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    return dir_name


def load_progress(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return json.load(f)
    return {}


def save_progress(file_path, progress):
    with open(file_path, 'w') as f:
        json.dump(progress, f)


def get_top250_books():
    book_links = []
    for page in range(0, 250, 25):
        url = f"https://book.douban.com/top250?start={page}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers)
        response.encoding = 'utf-8'
        soup = BeautifulSoup(response.text, 'html.parser')
        books = soup.find_all('div', class_='pl2')
        for book in books:
            link = book.find('a')['href']
            book_links.append(link)

    print(f"共爬取到 {len(book_links)} 本书的链接。")  # 显示爬取到的书籍数量
    return book_links


def scrape_comments(book_url, cookies):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36',
        'Accept-Language': 'zh-CN,zh;q=0.9'
    }
    comments = []
    page_num = 0
    previous_comments = []

    while True:
        page_num += 1
        comments_url = f"{book_url}comments/?start={(page_num - 1) * 20}&limit=20&status=P&sort=score"
        print(f"请求第 {page_num} 页评论: {comments_url}")

        res = requests.get(comments_url, cookies=cookies, headers=headers)
        if res.status_code != 200:
            print(f"请求失败，状态码: {res.status_code}")
            break

        soup = BeautifulSoup(res.text, 'html.parser')
        comment_list = soup.find_all('span', class_='short')

        # 检查当前页面是否没有评论
        if len(comment_list) == 0:
            print(f"第 {page_num} 页没有评论，停止请求。")
            break

        # 如果获取的评论数少于20，则认为已经没有更多评论
        if len(comment_list) < 20:
            print(f"获取到的评论数少于20条，停止请求，跳到下一本书。")
            break

        current_comments = [com.string.strip() for com in comment_list if com.string]

        # 检查当前评论是否与之前的评论相同
        if current_comments == previous_comments:
            print(f"第 {page_num} 页评论与上页相同，停止请求，跳到下一本书。")
            break

        comments.extend(current_comments)  # 添加新获取的评论
        previous_comments = current_comments  # 更新上一次的评论

        time.sleep(random.uniform(5, 15))

    return comments


def save_to_excel(book_url, comments, directory):
    book_id = book_url.split('/')[-2]
    file_path = os.path.join(directory, f"{book_id}_comments.xlsx")
    df = pd.DataFrame(comments, columns=['评论内容'])
    df.to_excel(file_path, index=False)


if __name__ == "__main__":
    cookies = {
        'cookie': 'bid=_HXRnJZ7t_g; dbcl2="283720641:efD1v6r5s08"; _pk_id.100001.3ac3=d042487526e67de9.1727623347.; __utmz=30149280.1727623348.1.1.utmcsr=accounts.douban.com|utmccn=(referral)|utmcmd=referral|utmcct=/; __utmz=81379588.1727623348.1.1.utmcsr=accounts.douban.com|utmccn=(referral)|utmcmd=referral|utmcct=/; push_noty_num=0; push_doumail_num=0; ck=-SfQ; __utma=30149280.1421786124.1727623348.1727623348.1727753953.2; __utmc=30149280; __utmt_douban=1; __utmb=30149280.1.10.1727753953; __utma=81379588.486489553.1727623348.1727623348.1727753953.2; __utmc=81379588; __utmt=1; __utmb=81379588.1.10.1727753953; ap_v=0,6.0; _pk_ref.100001.3ac3=%5B%22%22%2C%22%22%2C1727753954%2C%22https%3A%2F%2Faccounts.douban.com%2F%22%5D; _pk_ses.100001.3ac3=1'  # 请确保这个cookie是有效的
    }

    top250_books = get_top250_books()
    save_directory = create_directory('douban_comments')
    progress_file = 'progress.json'
    progress = load_progress(progress_file)
    total_books = 0

    for book_url in top250_books:
        if book_url in progress:
            print(f"{book_url} 已处理，跳过。")
            continue

        print(f"正在处理 {book_url} 的热门短评...")
        comments = scrape_comments(book_url, cookies)
        save_to_excel(book_url, comments, save_directory)
        progress[book_url] = len(comments)  # 保存爬取的评论数量
        total_books += 1
        save_progress(progress_file, progress)  # 保存进度
        print(f"{book_url} 爬取完成，评论已保存。")

    print(f"共爬取了 {total_books} 本书的评论。")
