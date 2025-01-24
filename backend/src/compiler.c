#include "../include/interface.h"


#include <sys/stat.h>
#include <sys/types.h>
#include <errno.h>
#include <unistd.h>
#include <limits.h>

//optimizations should happen at a graph level, 
//there should be a struct with every tensor that is being used.

//TODO move defs to header

void write_forward(tensor *root_node, char *file_name){
    (void) root_node; (void) file_name;
}

void write_backward(tensor *root_node, char *file_name){
    (void) root_node; (void) file_name;
}

void dynamic_compile(char *file_name) {
    char command[1024];
    char cwd[PATH_MAX];

    if (getcwd(cwd, sizeof(cwd)) == NULL) {
        perror("Error getting current working directory");
        return;
    }

    char abs_file_path[PATH_MAX];
    snprintf(abs_file_path, sizeof(abs_file_path), "%s/%s", cwd, file_name);

    snprintf(command, sizeof(command),
             "clang -o %s.out %s "
             "-L%s/.. -llightlemur -lm "
             "-L/opt/homebrew/opt/libomp/lib -lomp "
             "-I/opt/homebrew/opt/libomp/include -I%s/backend/include",
             abs_file_path, abs_file_path, cwd, cwd);
    // printf("%s\n",command);
    int result = system(command);

    if (result != 0) {
        fprintf(stderr, "Error: Compilation failed for file '%s'.\n", file_name);
    } else {
        printf("Compilation succeeded for file '%s'.\n", file_name);
    }
}

void create_directory_and_file(char *dir_name, char *file_name) {
    struct stat st = {0};
    if (stat(dir_name, &st) == -1) {
        if (mkdir(dir_name, 0755) == 0) {
            printf("Directory '%s' created successfully.\n", dir_name);
        } else {
            perror("Error creating directory");
        }
    }
    FILE *file = fopen(file_name, "w");
    if (file == NULL) {
        perror("Error creating file");
        return;
    }
    fprintf(file, "#include \"../backend/include/interface.h\"\n");
    fprintf(file, "\nint main(){\n\treturn 0;\n}\n");
    if (fclose(file) == 0) {
        printf("File '%s' created successfully.\n", file_name);
    } else {
        perror("Error closing file");
    }
}


void compile(tensor *root_node){

    //gets root node and generates a file named <root_node ptr val>.c in the compiled folder
    char *dir_name = "../lemurcompiled";
    char file_name[1024];
    snprintf(file_name, sizeof(file_name), "%s/%p.c", dir_name, (void *)root_node);
    create_directory_and_file(dir_name, file_name);

    write_forward(root_node, file_name);

    write_backward(root_node, file_name);

    dynamic_compile(file_name);
}

void compute(){

}