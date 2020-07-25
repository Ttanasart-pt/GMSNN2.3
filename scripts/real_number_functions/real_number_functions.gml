function lerp_float(from, to, speed){
	if(abs(from - to) < 0.001)
        return to;
    else
        return from + (to - from) / speed * delta_time/15000;
}

function in_range(val, from, to){
	return val > min(from, to) && val < max(from, to);
}