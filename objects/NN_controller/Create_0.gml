/// @description init
#region draw
	draw_set_circle_precision(128);
	fps_runner = 0;
	fps_show = fps_real;
#endregion

#region score
	point[0] = 0;
	point[1] = 0;
#endregion
#region NN games
	repeat(25){
		instance_create_depth(0, 0, 0, NN_game);	//create NN world
	}
	
	game_hover = -1;
	game_len = 21;
	
	NN_world_create(1);
#endregion

function NN_train(){
	#region save & mutate
		if(game_done.agent.__nn__fitness >= __nn__max_fitness[0]){
			NN_world_save_agent(game_done.agent, 0);
		}

		with(game_done.agent){
			NN_copy(other.__nn__player[0]);	
			NN_agent_mutate(.4);
			NN_agent_reset();
		}
	#endregion
	#region evolve
		__nn__offspring++;
		if(__nn__offspring > __nn__gen_size){
			NN_world_evolve();
		}
	#endregion
}