import json

def get_json_data_into_dict(json_file_name):

	with open(json_file_name) as f:
		data = f.read()

	dic = json.loads(data)

	return dic


if __name__ == '__main__':
	from config import config_path
	print(get_json_data_into_dict(json_file_name=config_path.JSON_USA_STATE_CODE))