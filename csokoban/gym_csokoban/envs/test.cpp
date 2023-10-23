#include "sokoban.h"
#include <iostream>
using namespace std;


string level_dir = "//home//sc//RS//thinker//csokoban//gym_csokoban//envs//boxoban-levels//unfiltered//train";
string img_dir = "//home//sc//RS//thinker//csokoban//gym_csokoban//envs//surface";

int main()
{
	cout << "size of sokoban: " << sizeof(Sokoban) << endl;
	Sokoban sokoban(false, level_dir, img_dir);
	unsigned char* obs = new unsigned char[sokoban.obs_n];
	float reward = 0.;
	bool done = false;
	sokoban.reset(obs);
	//sokoban.read_level(1002);
	sokoban.print_level();
	int a;
	while (cin >> a) {
		sokoban.step(a, obs, reward, done);
		sokoban.print_level();
		cout << "reward: " << reward << " done: " << done << endl;
	}
	delete[] obs;
}
