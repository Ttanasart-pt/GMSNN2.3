function NN_agent_create(in, hidden, out){
	#region data
		__nn__score = 0;
		__nn__avg_base = 0;
		__nn__fitness = 0;
	#endregion
	#region input nodes
		for(var i=0; i<in; i++){
			__nn__inputs[i] = 0;	
		}
	#endregion
	#region hidden nodes
		for(var a = 0; a < array_length(hidden); a++){
			var hidd_len = hidden[a];
			
			var weight = 0;
			var bias = 0;
			for(var i=0; i<hidd_len; i++){
				__nn__neurons[a][i] = 0;
				__nn__raw_neurons[a][i] = 0;
				var prev_len = a==0? in : hidden[a - 1];
		
				for(var j = 0; j < prev_len; j++){
					weight[i, j] = random_range(-1.0, 1.0);
					bias[i, j] = random_range(-1.0, 1.0);
				}
			}
			__nn__weights[a] = array_duplicate(weight);
			__nn__biases[a] = array_duplicate(bias);
		}
	#endregion
	#region output nodes
		for(var i = 0; i < out; i++){
			__nn__outputs[i] = 0;
	
			var len = hidden[array_length(hidden)-1];
			for(var j=0; j<len; j++){
				__nn__output_weights[i, j] = random_range(-1.0, 1.0);	
				__nn__output_bias[i, j] = random_range(-1.0, 1.0);
			}
		}
	#endregion
}

function NN_agent_forward(){
	#region input and hidden layers
		for(var a = 0; a < array_length(__nn__neurons); a++){
			var w = __nn__weights[a];
			var b = __nn__biases[a];
	
			for(var i = 0; i < array_length(__nn__neurons[a]); i++){
				__nn__neurons[a][i] = 0;
		
				if(a==0){
					for(var j = 0; j < array_length(__nn__inputs); j++){
						__nn__neurons[a][i] += __nn__inputs[j] * w[i][j] + b[i][j];	
					}
				} else {
					for(var j = 0; j < array_length(__nn__neurons[a - 1]); j++){
						__nn__neurons[a][i] += __nn__neurons[a-1][j] * w[i][j] + b[i][j];	
					}
				}
				
				__nn__raw_neurons[a][i] = __nn__neurons[a][i];
				__nn__neurons[a][i] = NN_activation(__nn__neurons[a][i]);
			}
		}
	#endregion
	#region output layer
		for(var i = 0; i < array_length(__nn__outputs); i++){
			__nn__outputs[i] = 0;
			var last_hid = array_length(__nn__neurons) - 1;
	
			for(var j = 0; j < array_length(__nn__neurons[last_hid]); j++){
				__nn__outputs[i] += __nn__neurons[last_hid][j] * __nn__output_weights[i][j] + __nn__output_bias[i][j];
			}
			
			__nn__outputs[i] = NN_activation(__nn__outputs[i]);
		}
	#endregion
}

function NN_agent_mutate(rate){
	#region hidden layer
		for(var a = 0; a < array_length(__nn__weights); a++){
			var w = __nn__weights[a];
			var b = __nn__biases[a];
	
			for(var i = 0; i < array_length(w); i++){
				for(var j = 0; j < array_length(w[i]); j++){
					w[i][j] = NN_mutate(w[i][j], rate);
					b[i][j] = NN_mutate(b[i][j], rate);
				}
			}
			__nn__weights[a] = w;
			__nn__biases[a] = b;
		}
	#endregion
	#region output
		for(var i = 0; i < array_length(__nn__output_weights); i++){
			for(var j = 0; j < array_length(__nn__output_weights[i]); j++){
				__nn__output_weights[i][j] = NN_mutate(__nn__output_weights[i][j], rate);
				__nn__output_bias[i][j] = NN_mutate(__nn__output_bias[i][j], rate);
			}
		}
	#endregion
}

function NN_agent_reset(){
	__nn__fitness = 0;
	__nn__avg_base = 0;
	__nn__score = 0;
}

function NN_agent_score(_score){
	__nn__score += _score;
	__nn__avg_base += 1;

	if(__nn__avg_base == 0)
		__nn__fitness = __nn__score
	else
		__nn__fitness = __nn__score / __nn__avg_base;
}
	
function NN_agent_backprop(rate, d_var){
	var layers = array_length(__nn__neurons);
	var cost, gradient, raw_gradient;
	var delta;

	/* Notes
	*	delta : dE/dy used in hidden layer propagation
	*/

	#region output layer
		/*		For each weight/bias in output layer
		*	1) Find error/cost function derivative (dE/dY from script cc_error_derivative)
		*	2) Find activation function derivative (dY/dy from script nn_acti_derivative)
		*	3) Find value-weight derivative (dy/dw = w in most case)
		*	4) Gradient = Multiply 1)*2)*3) (error/value derivative)
		*	5) Use gradient to recalculate weight/bias
		*	6) save gradient as delta matrix i = layers-1
		*/
		for(var o = 0; o < array_length(__nn__outputs); o++){
			for(var i = 0; i < array_length(__nn__output_weights); i++){
				var cost = NN_error(__nn__outputs[o], d_var[o]);
				
				for(var j = 0; j < array_length(__nn__output_weights[i]); j++){
					raw_gradient = NN_error_derivative(__nn__outputs[o]) 
								* NN_activation_derivative(__nn__neurons[layers-1][j])
								* cost;
					gradient = raw_gradient * __nn__neurons[layers-1][j];
					
					__nn__output_weights[i][j] = __nn__output_weights[i][j] - rate * gradient;
					if(global.__nn__weightclamp)
						__nn__output_weights[i][j] = clamp(__nn__output_weights[i][j], -1, 1)
					delta[layers-1][j] = raw_gradient * __nn__raw_neurons[layers-1][j];
				}
			}
		}
	#endregion

	#region hidden layers
		/*		For each weight/bias in hidden layer
		*	1) Find sum of delta * weight of the next layer
		*	2) Find delta on i=layer by nn_acti_derivative(raw_neuron) * 1)
		*	3) Gradient = 2) * (dy/dw : w)
		*	4) If there's more layer backward, save gradient as delta matrix i = a-1
		*/
		var sum_dw = 0;
	
		for(var a = layers - 1; a >= 0; a--){
			var w = __nn__weights[a];
			var b = __nn__biases[a];
			
			for(var i = 0; i < array_length(w); i++){
				for(var j = 0; j < array_length(w[i]); j++){
					sum_dw = 0;
					#region delta sum
						if(a == layers - 1){
							for(var k =0 ; k < array_length(__nn__outputs); k++){
								sum_dw += delta[a][k] * __nn__output_weights[k][i];
							}
						} else {
							var w_n = __nn__weights[a + 1];
							for(var k = 0; k < array_length(__nn__neurons[a + 1]); k++){
								sum_dw += delta[a][k] * w_n[k][i];
							}
						}
					#endregion
					
					gradient = __nn__neurons[a][i] * sum_dw;
					w[i][j] = w[i][j] - rate * gradient;
					if(global.__nn__weightclamp)
						w[i][j] = clamp(w[i][j], -1, 1)
				
					if(a > 0) 
						delta[a-1][i] = NN_activation_derivative(__nn__raw_neurons[a][i]) * sum_dw;
				}
			}
			
			__nn__weights[a] = w;
			__nn__biases[a] = b;
		}
	#endregion
}