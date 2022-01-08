from win32com.client import Dispatch, DispatchEx
import win32com
import datetime
import time
from bs4 import BeautifulSoup
import os

## TODO : Using a list of params_value


def run_excel_macro(sheet_path, sheet_macro_name, param_value = None):

	xlApp = Dispatch("Excel.Application")
	xlApp.Visible = True
	wb = xlApp.Workbooks.Open(sheet_path, ReadOnly=True)

	# Save the Range to HTML file in path
	file_name = datetime.datetime.now().strftime("%Y%m%d_%H%M_%f") + sheet_macro_name + ".htm"

	try:
		if param_value is None:
			xlApp.Application.Run(sheet_macro_name)
			wb.Close(True)
		else:
			xlApp.Application.Run(sheet_macro_name, param_value)
			wb.Close(True)

		xlApp.Application.Quit()
		wb = None
		xlApp = None
		del xlApp

	except Exception as e:
		print(e)
		xlApp.Application.Quit()
		wb = None
		xlApp = None
		del xlApp


def get_html_soup_from_excel_sheet_range(sheet_path, html_files_path, sheet_macro_name, param_value=None, param_value2=None, close_after=True):

	if not os.path.exists(html_files_path):
		print('Creating directory', html_files_path)
		os.mkdir(html_files_path)

	file_name = generate_html_file_from_excel_book(sheet_path=sheet_path,html_files_path=html_files_path,sheet_macro_name=sheet_macro_name,param_value=param_value,param_value2=param_value2,close_after=close_after)

	# Read the HTML File
	time.sleep(1)

	soup = retrieve_html_file_name(html_files_path=html_files_path,file_name=file_name)
	return soup


def generate_html_file_from_excel_book(sheet_path, html_files_path, sheet_macro_name, param_value=None, param_value2=None, close_after=True):

	xlApp = Dispatch("Excel.Application")
	xlApp.Visible = True
	wb = xlApp.Workbooks.Open(sheet_path, ReadOnly=True)
	# , UpdateLinks=False

	# Save the Range to HTML file in path
	file_name = datetime.datetime.now().strftime("%f") + ".htm"

	try:

		if param_value==None:
			xlApp.Application.Run(sheet_macro_name, html_files_path, file_name)
		else:
			if param_value2==None:
				xlApp.Application.Run(sheet_macro_name, html_files_path, file_name,param_value)
			else:
				xlApp.Application.Run(sheet_macro_name, html_files_path, file_name, param_value,param_value2)

		if close_after:
			# wb.Close(False)
			map(lambda book: book.Close(False), xlApp.Workbooks)
			xlApp.Quit()

		# xlApp.Application.Quit()
		wb = None
		xlApp = None
		del xlApp

	except Exception as e:
		print(e)
		# wb.Close(False)
		map(lambda book: book.Close(False), xlApp.Workbooks)
		xlApp.Quit()
		xlApp.Application.Quit()
		# del xlApp
		return file_name

	return file_name


def retrieve_html_file_name(html_files_path,file_name):

	if file_name is not None:
		full_path_file = html_files_path + "\\" + file_name
		soup = BeautifulSoup(open(full_path_file), "html.parser")

		if os.path.exists(full_path_file): os.remove(full_path_file)

		# Adjust the aligment to left
		div = soup.find(name='div')
		div['align'] = 'left'
	else:
		soup = BeautifulSoup()

	return soup


def save_images_from_spreadsheet(html_files_path ,sheet_path,sheet_macro_name, param_value):

	xlApp = Dispatch("Excel.Application")
	xlApp.Visible = True
	wb = xlApp.Workbooks.Open(sheet_path, ReadOnly=True)

	try:
		xlApp.Application.Run(sheet_macro_name, html_files_path, param_value)
		# wb.Close(False)
		# xlApp.Application.Quit()
		map(lambda book: book.Close(False), xlApp.Workbooks)
		xlApp.Quit()
		wb = None
		xlApp = None
		del xlApp

	except Exception as e:
		print(e)
		# wb.Close(True)
		# xlApp.Application.Quit()
		map(lambda book: book.Close(False), xlApp.Workbooks)
		xlApp.Quit()
		wb = None
		xlApp = None
		del xlApp


def close_all_excel_instances():
	import os
	os.system('taskkill /f /im Excel.exe')


def get_headers_from_worksheet(ws):
	import openpyxl
	res = [each.value for each in ws[1]]
	return res


def clear_range(ws,range_name):

	for row in ws[range_name]:
		for cell in row:
			cell.value = None


def get_multiple_html_soup_from_excel_sheet_range(sheet_path, html_files_path, sheet_macro_name, param_value=None, param_value2=None, close_after=True):

	file_name = generate_html_file_from_excel_book(sheet_path=sheet_path, html_files_path=html_files_path, sheet_macro_name=sheet_macro_name, param_value=param_value, param_value2=param_value2,
												   close_after=close_after)

	list_soup = []

	time.sleep(1)
	soup = retrieve_html_file_name(html_files_path=html_files_path, file_name=file_name)

	list_soup.append(soup)

	i=2
	while os.path.exists(html_files_path + "\\" + file_name.replace('.htm',f'_{i}.htm')):
		time.sleep(1)
		soup_sub = retrieve_html_file_name(html_files_path=html_files_path, file_name=file_name.replace('.htm',f'_{i}.htm'))
		list_soup.append(soup_sub)
		i += 1

	return list_soup


def spawn_excel_process_with_bbg_loaded():
	print('Starting Excel, loading bloomberg add ins')
	xlApp = DispatchEx("Excel.Application")
	xlApp.DisplayAlerts = False
	xlApp.Visible = True
	time.sleep(2)
	xlApp.Workbooks.Open(r'C:/blp/API/Office Tools/BloombergUI.xla') # Addin 2
	time.sleep(2)
	xlApp.RegisterXLL(r'C:/blp/API/office tools/bofaddin.dll') #Addin 1
	
	time.sleep(2)
	xlApp.WindowState = -4137
	return xlApp



def open_refresh_and_save_multiple_workbooks(workbook_paths):
	xlApp = spawn_excel_process_with_bbg_loaded()
	count = 0 
	start = time.time()
	
	for worbook in workbook_paths:
		try:
			print(f'Refreshing model: {worbook}')
			xlApp.AskToUpdateLinks = False 
			xlApp.DisplayAlerts = False
			wb = xlApp.Workbooks.Open(worbook, UpdateLinks=True, ReadOnly=False)
			if xlApp.WindowState != -4137:
				xlApp.WindowState = -4137 # Center window, better when debugging
			
			if wb.ReadOnly : 
				print(worbook, 'READ ONLY - skipping')
				wb.Close(False)  # Close and don't save
				continue
			
			xlApp.Calculation = -4135 # Manual Calc avoid getting in an infinite loop
			wb.RefreshAll()
			sheets = [sh.Name for sh in wb.Sheets]
			
			
			# Special resiliency behaviour for SB models, else generic attempt
			if {'Drivers', 'Valuation', 'Model'}.intersection(set(sheets)) == {'Drivers', 'Valuation', 'Model'}:
				try:
					ws = wb.Worksheets('Drivers')
					formula = '=IFERROR(COUNTIF(Drivers!$D$1:$AZ$2000,"*Requesting*"),0)+IFERROR(COUNTIF(Valuation!$D$1:$AZ$2000,"*Requesting*"),0)+IFERROR(COUNTIF(Model!$D$1:$AZ$2000,"*Requesting*"),0)'
					ws.Cells.Range("C1").Formula = formula
				except Exception as E:
					print(E)
			
			else:
				ws = wb.ActiveSheet
				formula = f"=SUMPRODUCT(--ISERROR('{sheets[0]}'!$B$1:$AZ$500))"
				if len(sheets) > 1 :
					for i,sheet in enumerate(sheets[1:]):
						formula += f"+SUMPRODUCT(--ISERROR('{sheet}'!$B$1:$AZ$500))"
						if i == 4 : break
				if ws.Name not in sheets[0:4]:
					formula += f"+SUMPRODUCT(--ISERROR('{ws.Name}'!$B$1:$AZ$500))" # Make sure we include active sheet in formula
				ws.Cells.Range("C1").Formula = formula
				
			pre_calculation_occurences = ws.Cells.Range("C1").Value
						
			iters = 0
			xlApp.RTD.RefreshData() # Refresh feeds
			for i in range(0,25):
				xlApp.Calculate()
				
			while iters < 40 and (ws.Cells.Range("C1").Value != pre_calculation_occurences and ws.Cells.Range("C1").Value !=0) :
				xlApp.Calculate()
				time.sleep(0.5)
				iters += 1
			print(f'Exited after {iters} additional iterations')
			xlApp.CalculateUntilAsyncQueriesDone()
			time.sleep(2)
			ws.Cells.Range("C1").Clear()
			wb.Save()
			wb.Close(True) # Close and save
			count += 1
			print(f'Done, average time per sheet: {round((time.time()-start)/count,2)} seconds')
		except Exception as E:
			print(E)
			try:
				wb.Close(False)
			except:
				pass
			xlApp.Quit()
			del xlApp
			xlApp = spawn_excel_process_with_bbg_loaded()
			continue
	
	xlApp.Quit()
	del xlApp
	end = time.time()
	print(f'Total runtime: {round(end-start,2)} seconds')
	

if __name__ == '__main__':
	from os import listdir
	from os.path import isfile, join
	DIR = r'S:/Intern/Oscar/Models Test/'
	files = r'S:/Intern/Oscar/Models Test/'
	onlyfiles = [DIR+f for f in listdir(DIR) if isfile(join(DIR, f)) if 'xls' in f]
	
	files = r'S:/Intern/Oscar/Models Test/ALSN Model.xlsx'
	
	risk_factor = [r'C:/Users/oscar.levy/Documents/Sandbar_Risk_Factor_Created.xlsm', r'C:/Users/oscar.levy/Documents/Sandbar_Risk_Factor_Created2.xlsm']
	open_refresh_and_save_multiple_workbooks(onlyfiles)
