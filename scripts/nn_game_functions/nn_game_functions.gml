function game_get_point(game){
	with(game.agent){
		agent_reset();
	}
	
	game.game_round++;
	if(game.game_round >= NN_controller.game_len){
		game_finish(game);	
	}
}

function game_finish(game){
	game.game_round = 0;
	with(NN_controller){
		game_done = game;
		NN_train();
	}
}