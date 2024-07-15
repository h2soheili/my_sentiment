import finpy_tse as fpy

if __name__ == '__main__':
    # https://pypi.org/project/finpy-tse/
    # pip install finpy-tse
    # finpy-tse~=1.2.10
    """
    line 2391
    fix:
      if type(code_df) == bool:
         continue
    """
    a = fpy.Build_Market_StockList(
        bourse=True,
        farabourse=True,
        payeh=True,
        detailed_list=True,
        show_progress=False,
        save_excel=False,
        save_csv=True,
        save_path='./stocks.csv')
    a.to_csv('./stocks.csv')
    # print(a)
