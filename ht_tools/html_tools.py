# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 21:31:49 2021

@author: Hogan
"""

import base64
import gc
import io


def color_negative_red(val):
	color = 'red' if val < 0 else 'black'
	return 'color: %s' % color


def get_html_from_df_and_css(df, css, css_class_name):
	html_string = """
		<html>
		<head><title>HTML Pandas Dataframe with CSS</title></head>
		<style>
			{css}
		</style>
		<body>
			{table}
		</body>
		</html>
	"""

	from sam_tools import file_tools
	css = file_tools.read_file(css)

	html = html_string.format(table=df.to_html(classes=css_class_name), css=css)

	return html


def matplotlib_fig_to_html(fig):
    bytes_temporary_allocation = io.BytesIO()
    fig.savefig(bytes_temporary_allocation, format='jpg')
    bytes_temporary_allocation.seek(0)
    base64_string = base64.b64encode(bytes_temporary_allocation.read())
    del bytes_temporary_allocation
    gc.collect()
    return f'<img src="data:image/png;base64,{base64_string.decode("ascii")}" alt="", style="text-align: center;" />'


def add_css_to_html(html, css_path):
	from bs4 import BeautifulSoup
	soup = BeautifulSoup(html, "html.parser")

	from sam_tools import file_tools
	css = file_tools.read_file(css_path)

	# existing_style = soup.find(name='style').text
	# new_style = '{existing}\n\n {new}'.format(existing=soup.find(name='style').text,new=css)

	div = soup.find(name='style')
	div.append(css)

	html = soup.prettify()
	return html