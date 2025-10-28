#include <iostream>

#include "stereo_system.hpp"
#include "file_parsing.hpp"

int main() {
    StereoSystem ss = deserializeStereoSystem(readFromFile("../assets/launchMonitorData.xml"));

    std::cout << "hello world" << std::endl;

    return 0;
}