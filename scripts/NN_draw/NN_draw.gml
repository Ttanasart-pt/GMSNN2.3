function NN_draw(xx, yy, target, align){
	var len, layers;
	layers = 2 + array_length(target.__nn__neurons);
	
	var node_w = 64;
	var node_h = 48;
		
	#region get size
		var ww = node_w * (layers - 1);
		var hh = max(array_length(target.__nn__inputs), array_length(target.__nn__outputs));
		for(i = 0; i < array_length(target.__nn__neurons); i++){
			if(array_length(target.__nn__neurons[i]) > hh) hh = array_length(target.__nn__neurons[i]);
		}
		hh = (hh-1) * node_h;
	#endregion
	#region get coordinate
		switch(align){
			case nn_draw_alignment.top_left:
				var sx1 = xx;	var sx2 = sx1 + ww;	
				var sy1 = yy;	var sy2 = sy1 + hh;	
				break;
			case nn_draw_alignment.top_right:
				var sx2 = xx;	var sx1 = sx2 - ww;
				var sy1 = yy;	var sy2 = sy1 + hh;	
				break;
			case nn_draw_alignment.bottom_left:
				var sx1 = xx;	var sx2 = sx1 + ww;	
				var sy2 = yy;	var sy1 = sy2 - hh;
				 break;
			case nn_draw_alignment.bottom_right:
				var sx2 = xx;	var sx1 = sx2 - ww;
				var sy2 = yy;	var sy1 = sy2 - hh;
				 break;
		}
	
		var syc = (sy1+sy2)/2;
	#endregion

	#region draw nodes
		var sys;
		for(i = 0; i < layers; i++){
			if(i == 0) 
				len = array_length(target.__nn__inputs);
			else if(i == layers - 1) 
				len = array_length(target.__nn__outputs);
			else  
				len = array_length(target.__nn__neurons[i - 1]);
			
			sys[i] = syc - (len - 1) / 2 * node_h;
			
			for(var j = 0; j < len; j++){
				draw_circle(sx1+node_w*i, sys[i]+node_h*j, 10, false);
			}
		}
	#endregion	
	#region draw __nn__weights
		draw_set_alpha(.3);
		var w, _w;
		for(var a = 0; a < layers - 1; a++){
			if(a == layers - 2){
				for(var i = 0; i < array_length(target.__nn__output_weights); i++){
					for(var j = 0; j < array_length(target.__nn__output_weights[i]); j++){
						w = NN_activation(target.__nn__output_weights[i][j]);
						draw_line_width(sx1 + node_w * a, sys[a] + node_h * j, 
							sx1 + node_w * (a + 1), sys[a + 1] + node_h * i, 
							w * 10);
					}
				}
			} else {
				w = target.__nn__weights[a];
				for(var i = 0; i < array_length(w); i++){
					len = a==0? array_length(target.__nn__inputs) : array_length(target.__nn__neurons[a-1]);
					
					for(var j = 0; j < len; j++){
						_w = NN_activation(w[i][j]);
						draw_line_width(sx1 + node_w * a, sys[a] + node_h * j,
							sx1 + node_w * (a + 1), sys[a + 1] + node_h * i, 
							_w * 10);
					}
				}
			}
		}
		draw_set_alpha(1);
	#endregion
}