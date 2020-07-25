function NN_activation(_x){
	var _e = 2.71828182;
	var e_x = power(_e, _x);
	var e_ix = 1 / e_x;
	
	switch(global.__nn__acti){
		case nn_acti_type.tanh:
			return (e_x-e_ix) / (e_x+e_ix);
		
		case nn_acti_type.softmax:
			//var soft_sum = 0;
			//for( var i=0; i<array_length_1d(all_x); i++){
			//	soft_sum += power(_e, all_x[i]);	
			//}
			//return e_x / soft_sum;
		
		case nn_acti_type.sigmoid:
			return e_x / (e_x+1);
	}
}

function NN_activation_derivative(_x){
	switch(global.__nn__acti){
		case nn_acti_type.tanh:
			return 1 - power(NN_activation(_x), 2);
		case nn_acti_type.softmax:
			return _x / (-_x+1);
		case nn_acti_type.sigmoid:
			var _ax = NN_activation(_x);
			return _ax * (1-_ax);
	}
}

function NN_error(val, d_val){
	switch(global.__nn__erro){
		case nn_error_type.different:
			return d_val - val;
		case nn_error_type.cross_entropy_loss:
			return val == 1? -ln(val) : -ln(1 - val);
	}
}

function NN_error_derivative(val){
	switch(global.__nn__erro){
		case nn_error_type.different:
			return -1;
		case nn_error_type.cross_entropy_loss:
			return val == 1? -1 / val : 1 / (1 - val);
	}
}

function NN_normalize(val, _min, _max){
	return clamp(((val - _min) / (_max - _min) * 2) - 1, -1, 1);
}

function NN_mutate(val, rate){
	var chan = random(100);

	if(chan < 10*rate){
		val *= -1;	
	} else if(chan < 20*rate){
		val = random_range(-1, 1);
	} else if(chan < 45*rate){
		val = random_range(val, val*2);
	} else if(chan < 70*rate){
		val = random_range(0, val);
	} 

	return global.__nn__weightclamp? clamp(val, -1, 1) : val;
}

function NN_copy(agent){
	__nn__weights			= agent.__nn__weights;
	__nn__biases			= agent.__nn__biases;
	__nn__output_weights	= agent.__nn__output_weights;
	__nn__output_bias		= agent.__nn__output_bias;
}