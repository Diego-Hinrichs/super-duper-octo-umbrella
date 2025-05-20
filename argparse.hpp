#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <sstream>
#include <stdexcept>
#include <iostream>
#include <functional>
#include <any>
#include <optional>

class ArgumentParser {
private:
    std::string program_name;
    std::unordered_map<std::string, std::string> help_messages;
    std::unordered_map<std::string, std::any> parsed_args;
    std::unordered_map<std::string, std::function<void(const std::string&)>> parsers;
    std::unordered_map<std::string, std::any> default_values;
    std::vector<std::string> positional_args;
    std::vector<std::string> flag_args;

public:
    ArgumentParser(const std::string& name = "") : program_name(name) {}

    template<typename T>
    void add_argument(const std::string& name, const std::string& help, const T& default_value) {
        help_messages[name] = help;
        default_values[name] = default_value;
        parsed_args[name] = default_value;

        parsers[name] = [this, name](const std::string& value) {
            if constexpr (std::is_same_v<T, int>) {
                parsed_args[name] = std::stoi(value);
            } else if constexpr (std::is_same_v<T, double>) {
                parsed_args[name] = std::stod(value);
            } else if constexpr (std::is_same_v<T, float>) {
                parsed_args[name] = std::stof(value);
            } else if constexpr (std::is_same_v<T, std::string>) {
                parsed_args[name] = value;
            } else if constexpr (std::is_same_v<T, bool>) {
                if (value == "true" || value == "1" || value == "yes" || value == "y") {
                    parsed_args[name] = true;
                } else if (value == "false" || value == "0" || value == "no" || value == "n") {
                    parsed_args[name] = false;
                } else {
                    throw std::invalid_argument("Invalid boolean value: " + value);
                }
            } else {
                throw std::invalid_argument("Unsupported type for argument: " + name);
            }
        };
    }

    void add_flag(const std::string& name, const std::string& help) {
        help_messages[name] = help;
        default_values[name] = false;
        parsed_args[name] = false;
        flag_args.push_back(name);
    }

    void parse_args(int argc, char* argv[]) {
        program_name = argv[0];
        
        for (int i = 1; i < argc; i++) {
            std::string arg = argv[i];
            
            // Check if it's a flag or an option
            if (arg.substr(0, 1) == "-") {
                std::string key = arg;
                if (key.substr(0, 2) == "--") {
                    key = key.substr(2);
                } else {
                    key = key.substr(1);
                }
                
                // Check if it's a flag
                bool is_flag = false;
                for (const auto& flag : flag_args) {
                    if (flag == key) {
                        is_flag = true;
                        parsed_args[key] = true;
                        break;
                    }
                }
                
                // If not a flag, it's an option with a value
                if (!is_flag) {
                    if (i + 1 >= argc) {
                        throw std::invalid_argument("Missing value for argument: " + key);
                    }
                    
                    std::string value = argv[++i];
                    
                    if (parsers.find(key) != parsers.end()) {
                        parsers[key](value);
                    } else {
                        throw std::invalid_argument("Unknown argument: " + key);
                    }
                }
            } else {
                // It's a positional argument
                positional_args.push_back(arg);
            }
        }
    }

    template<typename T>
    T get(const std::string& name) const {
        if (parsed_args.find(name) == parsed_args.end()) {
            throw std::invalid_argument("Argument not found: " + name);
        }
        
        try {
            return std::any_cast<T>(parsed_args.at(name));
        } catch (const std::bad_any_cast&) {
            throw std::invalid_argument("Type mismatch for argument: " + name);
        }
    }

    void print_help() const {
        std::cout << "Usage: " << program_name << " [options]" << std::endl;
        std::cout << "Options:" << std::endl;
        
        for (const auto& [key, help] : help_messages) {
            std::cout << "  -" << key;
            
            // Check if it's a flag
            bool is_flag = false;
            for (const auto& flag : flag_args) {
                if (flag == key) {
                    is_flag = true;
                    break;
                }
            }
            
            if (!is_flag) {
                std::cout << " <value>";
            }
            
            std::cout << "\t" << help;
            
            // Print default value if not a flag
            if (!is_flag && default_values.find(key) != default_values.end()) {
                std::cout << " (default: ";
                
                const auto& default_value = default_values.at(key);
                
                if (default_value.type() == typeid(int)) {
                    std::cout << std::any_cast<int>(default_value);
                } else if (default_value.type() == typeid(double)) {
                    std::cout << std::any_cast<double>(default_value);
                } else if (default_value.type() == typeid(float)) {
                    std::cout << std::any_cast<float>(default_value);
                } else if (default_value.type() == typeid(std::string)) {
                    std::cout << std::any_cast<std::string>(default_value);
                } else if (default_value.type() == typeid(bool)) {
                    std::cout << (std::any_cast<bool>(default_value) ? "true" : "false");
                }
                
                std::cout << ")";
            }
            
            std::cout << std::endl;
        }
    }
};
