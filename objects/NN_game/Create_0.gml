/// @description init
#region data
	playing = true;
	agent = instance_create_depth(0, 0, -100, NN_game_agent);
	agent.game = id;
	
	game_round = 0;
#endregion