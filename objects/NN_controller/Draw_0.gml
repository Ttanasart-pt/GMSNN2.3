/// @description init
#region score
	draw_set_halign(fa_left);
	draw_set_valign(fa_top);
	
	max_fit = -999;
	fitt_index = 0;
	
	for(i = 0; i < instance_number(NN_game); i++){
		var game = instance_find(NN_game, i);
		if(game.agent.__nn__fitness > max_fit){
			max_fit = game.agent.__nn__fitness;
			fitt_index = i;
		}
	}
	
	game_hover = noone;
	
	for(i = 0; i < instance_number(NN_game); i++){
		game = instance_find(NN_game, i);
		if(game == __nn__fittest[0]){
			draw_set_color(c_lime);
		} else if(i == fitt_index) {
			draw_set_color(c_yellow);
		} else {
			draw_set_color(c_white);
		}
		
		if(in_range(mouse_x, 16, 100) && in_range(mouse_y, 16+i*24, 16+(i+1)*24)){
			game_hover = game;	
			draw_set_alpha(0.1);
			draw_rectangle(16, 16+i*24, 100, 16+(i+1)*24, false);
			draw_set_alpha(1);
			
			#region force evolve
				if(mouse_check_button_pressed(mb_left)){
					__nn__fittest[0] = game;
					__nn__fittest[1] = game;
	
					NN_copy(game.agent);
					NN_world_evolve();
				}
			#endregion
		}
		
		draw_set_font(f_ui_12);
		draw_text(48, 16 + i * 24, string(game.agent.__nn__fitness));
		draw_set_font(f_ui_12b);
		draw_text(16, 16 + i * 24, string(game.game_round) + " : ");
	}	
	
	draw_set_color(c_white);
	draw_set_halign(fa_right);
	draw_set_valign(fa_top);
	draw_set_font(f_ui_16b);
	draw_text(room_width - 16, 16, "Fittest " + string(__nn__max_fitness[0]));
	draw_set_font(f_ui_12);
	draw_text(room_width - 16, 48, "Gen #" + string(__nn__generation) + "/" + string(__nn__offspring));
	
	draw_set_halign(fa_left);
	draw_set_valign(fa_bottom);
	draw_set_font(f_ui_12);
	draw_text(16, room_height - 16, "fps " + string(fps_show));
	if(fps_runner++ > 10){
		fps_runner = 0;
		fps_show = fps_real;
	}
#endregion

#region NN
	draw_set_color(c_white);
	var p;
	for(var i = 0; i < __nn__players; i++){
		if(__nn__fittest[i] != noone || game_hover != noone){
			if(game_hover == noone)	p = __nn__fittest[i].agent;
			else					p = game_hover.agent;
			NN_draw(room_width - 32 - i * 300, room_height - 32, 
				p, nn_draw_alignment.bottom_right);
		}
	}
#endregion