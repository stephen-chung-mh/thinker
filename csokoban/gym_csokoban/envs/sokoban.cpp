#include "sokoban.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdio.h>
#include <cstdlib>
#include <stdexcept>
#include <random>
#include <cassert>

constexpr float reward_step = -0.01f, reward_box_off = -1.f, reward_box_on = +1.f, reward_finish = +10.f;
constexpr int max_step_n = 120;

roomStatus char_to_roomStatus(const char c) {
	switch (c)
	{
	case '#':
		return roomStatus::wall;
	case ' ':
		return roomStatus::empty;
	case '$':
		return roomStatus::box_not_on_tar;
	case '!':
		return roomStatus::box_on_tar;
	case '.':
		return roomStatus::tar;
	case '@':
		return roomStatus::player_not_on_tar;
	case '%':
		return roomStatus::player_on_tar;
	}
	return roomStatus::wall;
}

char roomStatus_to_char(const roomStatus r) {
	switch (r)
	{
	case roomStatus::wall:
		return '#';
	case roomStatus::empty:
		return ' ';
	case roomStatus::box_not_on_tar:
		return '$';
	case roomStatus::box_on_tar:
		return '!';
	case roomStatus::tar:
		return '.';
	case roomStatus::player_not_on_tar:
		return '@';
	case roomStatus::player_on_tar:
		return '%';
	}
	return '#';
}


void read_bmp(const string &img_dir, const string &img_name, vector<unsigned char> &data)
{
	string full_path = img_dir + "//" + img_name;
	ifstream in(full_path.c_str(), ios::in | ios::binary);
	if (in.is_open())
	{
		char cwidth[4], cheight[4];
		in.seekg(18);
		in.read(cwidth, 4);
		in.read(cheight, 4);
		int height = *((int*)cheight);
		int width = *((int*)cwidth);
		int padded_width = (width + 3) & (~3);
		//cout << "height: " << height << " width: " << padded_width << endl;
		char* tmp = new char[width * 3];
		for (int y = height - 1; y >= 0; y--) {
			in.seekg(54 + y * padded_width * 3);
			in.read(tmp, width * 3);
			for (int x = 0; x < width; x++) {
				data.push_back(tmp[x * 3 + 2]);
				data.push_back(tmp[x * 3 + 1]);
				data.push_back(tmp[x * 3]);
			}
		}
		delete[] tmp;
		in.close();
	}
	else {
		throw ios_base::failure("unable to open " + full_path);
	}
	return;
}

void Sokoban::read_spirits() {
	string s = small ? "_small.bmp" : ".bmp";
	read_bmp(img_dir, "wall" + s, spirites[0]);
	read_bmp(img_dir, "floor" + s, spirites[1]);
	read_bmp(img_dir, "box" + s, spirites[2]);
	read_bmp(img_dir, "box_on_target" + s, spirites[3]);
	read_bmp(img_dir, "player" + s, spirites[4]);
	read_bmp(img_dir, "player_on_target" + s, spirites[5]);
	read_bmp(img_dir, "box_target" + s, spirites[6]);
	defEngine = std::mt19937(seed);
}

void Sokoban::move_pos(const action a, int& x, int& y) {
	switch (a)
	{
	case action::up:
		y--;
		return;
	case action::down:
		y++;
		return;
	case action::left:
		x--;
		return;
	case action::right:
		x++;
		return;
	case action::noop:
		return;
	}
}

void Sokoban::move_player(roomStatus& old_r, roomStatus& new_r) {
	if (old_r == roomStatus::player_on_tar)
		old_r = roomStatus::tar;
	else
		old_r = roomStatus::empty;
	if (new_r == roomStatus::tar)
		new_r = roomStatus::player_on_tar;
	else
		new_r = roomStatus::player_not_on_tar;
	return;
}

float Sokoban::move_box(roomStatus& old_r, roomStatus& new_r) {
	float reward = 0.;
	if (old_r == roomStatus::box_on_tar)
	{
		old_r = roomStatus::tar;
		reward += reward_box_off;
		box_left++;
	}
	else
		old_r = roomStatus::empty;
	if (new_r == roomStatus::tar) {
		new_r = roomStatus::box_on_tar;
		reward += reward_box_on;
		box_left--;
		if (box_left == 0) {
			reward += reward_finish;
			done = true;
		}
	}
	else
		new_r = roomStatus::box_not_on_tar;
	return reward;
}

float Sokoban::move(const action a) {
	if (done) {
		return 0.;
	}
	else if (a == action::noop) {
		return reward_step;	
	}
	else{
		roomStatus& old_r = room_status[player_pos_y][player_pos_x];
		int new_pos_x = player_pos_x, new_pos_y = player_pos_y;
		move_pos(a, new_pos_x, new_pos_y);
		if (new_pos_x < 0 || new_pos_x >= room_x || new_pos_y < 0 || new_pos_y >= room_y){
			return reward_step;
		}
		else{
			roomStatus& new_r = room_status[new_pos_y][new_pos_x];
			if (new_r == roomStatus::wall) {
				return reward_step;
			} 
			else if (new_r == roomStatus::empty || new_r == roomStatus::tar)
			{
				move_player(old_r, new_r);
				player_pos_x = new_pos_x;
				player_pos_y = new_pos_y;
				return reward_step;
			}
			else if (new_r == roomStatus::box_not_on_tar || new_r == roomStatus::box_on_tar)
			{
				int new_box_pos_x = new_pos_x, new_box_pos_y = new_pos_y;
				move_pos(a, new_box_pos_x, new_box_pos_y);
				if (new_box_pos_x < 0 || new_box_pos_x >= room_x || new_box_pos_y < 0 || new_box_pos_y >= room_y) {
					return reward_step;
					}
				else{
					roomStatus& new_box_r = room_status[new_box_pos_y][new_box_pos_x];
					if (new_box_r == roomStatus::empty || new_box_r == roomStatus::tar) {
						float reward = move_box(new_r, new_box_r);
						move_player(old_r, new_r);
						player_pos_x = new_pos_x;
						player_pos_y = new_pos_y;
						return reward_step + reward;
			 		} else {
						return reward_step;
					}
				}
			}
			else {
				return reward_step;
			}
		}
	}
}

int Sokoban::print_level() {
	for (const auto& row : room_status) {
		for (const auto& col : row)
			cout << roomStatus_to_char(col);
		cout << endl;
	}
	cout << "player pos: (" << player_pos_x << "," << player_pos_y << ")" << endl;
	return 0;
}

int Sokoban::read_level(const int room_id)
{
	box_left = 0;
	done = false;

	char file_name[10];
	snprintf(file_name, 10, "%03d.txt", room_id / 1000);
	string full_path = level_dir + "//" + file_name;
	//std::cout << "reading from " << full_path << " level " << room_id << endl;
	ifstream in(full_path.c_str(), ios::in);
	if (in.is_open())
	{
		string line;
		int n = 0, m = (room_id % 1000) * 12 + 1;
		while (getline(in, line) && n < m + room_y)
		{
			if (n++ >= m) {
				istringstream row(line);
				char c;
				int x = 0;
				while (row.get(c) && c != '\n')
				{
					roomStatus r = char_to_roomStatus(c);
					room_status[n - m - 1][x] = r;
					if (r == roomStatus::player_not_on_tar || r == roomStatus::player_on_tar) {
						player_pos_x = x;
						player_pos_y = n - m - 1;
					}
					else if (r == roomStatus::box_not_on_tar) box_left++;
					x++;
				}
			}
		};
		in.close();
	}
    if (box_left != 4) {
        std::ostringstream error_msg;
        error_msg << "box_left must be equal to 4 (room_id: " << room_id << ")";
        throw std::runtime_error(error_msg.str());
    }
	return 0;
}

void Sokoban::render(unsigned char* obs) {
	if (small)
		for (int y = 0; y < room_y; y++)
			for (int x = 0; x < room_x; x++)
				for (int sy = 0; sy < small_img_y; sy++)
					for (int sx = 0; sx < small_img_x; sx++)
						for (int d = 0; d < 3; d++)
							obs[((y * small_img_y + sy) * (room_x * small_img_x) + (x * small_img_x + sx)) * 3 + d] = spirites[int(room_status[y][x])][(sy * small_img_y + sx) * 3 + d];
	else
		for (int y = 0; y < room_y; y++)
			for (int x = 0; x < room_x; x++)
				for (int sy = 0; sy < large_img_y; sy++)
					for (int sx = 0; sx < large_img_x; sx++)
						for (int d = 0; d < 3; d++)
							obs[((y * large_img_y + sy) * (room_x * large_img_x) + (x * large_img_x + sx)) * 3 + d] = spirites[int(room_status[y][x])][(sy * large_img_y + sx) * 3 + d];

	/*
	ofstream out("debug.ppm", ios::out);
	out << "P3\n" << to_string(room_x * img_x) << " " << to_string(room_y * img_y) << " \n255 \n";
	for (int y = 0; y < room_y * img_y; y++){
		for (int x = 0; x < room_x * img_x; x++) {
			int ind = (y * (room_x * img_x) + x) * 3;
			out << to_string(obs[ind]) << " " << to_string(obs[ind + 1]) << " " << to_string(obs[ind + 2]) << " ";
		}
		out << endl;
	}
	out.close();*/
}

void Sokoban::reset(unsigned char* obs) {	
	uniform_int_distribution<int> roomDist(0, level_num - 1);
	uniform_int_distribution<int> stepDist(0, 5);
	int room_id = roomDist(defEngine);	
	// std::cout << "sampling from " << level_num * 1000 - 1 << " and get " << room_id << " seed : " << seed << endl;
	read_level(room_id);
	step_n = stepDist(defEngine);
	render(obs);
}

void Sokoban::reset_level(unsigned char* obs, const int room_id) {
	read_level(room_id);
	uniform_int_distribution<int> stepDist(0, 5);
	step_n = stepDist(defEngine);
	render(obs);
}

void Sokoban::step(const action a, unsigned char* obs, float& reward, bool& done, bool& truncated_done) {
	reward = move(a);		
	if (step_n >= max_step_n - 1) {
		this->done = true;
		truncated_done = true;
	}
	else {
		step_n++;	
		truncated_done = false;
	}
	done = this->done;
	render(obs);
}

void Sokoban::step(const int a, unsigned char* obs, float& reward, bool& done, bool& truncated_done) {
	if (a >= 0 && a <= 4) step(action(a), obs, reward, done, truncated_done);
	else throw invalid_argument("invalid action");
}

void Sokoban::clone_state(unsigned char* room_status, int& step_n, bool& done) {
	step_n = this->step_n;
	done = this->done;
	for (int y = 0; y < room_y; y++)
		for (int x = 0; x < room_x; x++)
			room_status[y * room_x + x] = (unsigned char)this->room_status[y][x];
}

void Sokoban::restore_state(const unsigned char* room_status, const int& step_n, const bool& done) {
	this->step_n = step_n;
	this->done = done;
	box_left = 0;
	for (int y = 0; y < room_y; y++)
		for (int x = 0; x < room_x; x++) {
			roomStatus r = (roomStatus)room_status[y * room_x + x];
			this->room_status[y][x] = r;
			if (r == roomStatus::player_not_on_tar || r == roomStatus::player_on_tar) {
				player_pos_y = y;
				player_pos_x = x;
			}
			else if (r == roomStatus::box_not_on_tar) {
				box_left++;
			}

		}
}

void Sokoban::set_seed(unsigned int seed){
	this->seed = seed;
	defEngine = std::mt19937(seed);
	//std::cout << "seed is called to set to " << seed << endl;
}