function ds_list_exist(list, member){
	return ds_list_find_index(list, member) != -1;
}

function array_duplicate(arr){
	var _arr = 0;
	for(var i = 0; i < array_length(arr); i++)
		for(var j = 0; j < array_length(arr[i]); j++)
			_arr[i][j] = arr[i][j];
	return _arr;
}