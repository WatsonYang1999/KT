import time
import os
def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        print(f"函数 {func.__name__} 开始执行")

        result = func(*args, **kwargs)

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"函数 {func.__name__} 执行结束")
        print(f"总共执行时间: {elapsed_time:.4f} 秒")

        return result

    return wrapper

def suit_directory_decorator():
    def decorator(func):
        def wrapper(*args, **kwargs):
            print_dir = False

            # 保存当前工作目录
            original_directory = os.getcwd()

            try:
                # 切换到指定目录
                while os.path.split(os.getcwd())[-1] != 'KT-Refine':
                    os.chdir('..')

                # 在新目录下执行工作
                if print_dir : print(f'当前工作目录: {os.getcwd()}')
                result = func(*args, **kwargs)
                return result

            finally:
                # 无论如何，在函数结束时切换回原始工作目录
                os.chdir(original_directory)
                if print_dir : print(f'函数结束，恢复到原始工作目录: {os.getcwd()}')

        return wrapper

    return decorator
