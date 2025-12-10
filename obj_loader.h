#ifndef OBJ_LOADER_H
#define OBJ_LOADER_H

#include "model.h"
#include <string>
#include <vector>

// Load an OBJ file and return a model
// Returns an empty model if loading fails
model load_obj_file(const std::string& file_path);

#endif // OBJ_LOADER_H
