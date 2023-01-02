import requests
import numpy as np
from bs4 import BeautifulSoup
import openpyxl as op


def scraping():

      year_list = np.arange(1992,2022,1)
      for year in year_list:
            res = requests.get(f"https://www.data.jma.go.jp/obd/stats/etrn/view/mb5daily_a1.php?prec_no=32&block_no=1167&year={year}&month=&day=&view=")
            soup = BeautifulSoup(res.text,"html.parser")
            tags = soup.find_all("td",class_="data_0_0")
            text_list = [x.string for x in tags]
            pre_list = [text_list[20*n+1] for n in range(72)]
            pre_array = np.array(pre_list)
            np.save(f"./cap_stone/precipitation_amount_{year}.npy",pre_array)

def write_xlsx():

      book = op.load_workbook(filename="./cap_stone/yoroibata.xlsx")
      sheet = book.worksheets[0]

      year_list = np.arange(1992,2022,1)
      column_list = np.arange(2,32,1)
      row_list = np.arange(3,75,1)
      for year,column in zip(year_list,column_list) :
            value_list = np.load(f"./cap_stone/precipitation_amount_{year}.npy")
            for val,row in zip(value_list,row_list) :
                  sheet.cell(row=row,column=column).value = val
      
      book.save("./cap_stone/yoroibata.xlsx")
      book.close()

write_xlsx()


