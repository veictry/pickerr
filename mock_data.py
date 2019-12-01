from faker import Faker
import random

def generate_simple_time_text_log():
    N = 100

    def generate_simple_log(target_path, N):
        for i in range(N):
            print(i)
            text_fake = Faker()
            time_fake = Faker()
            text_fake.seed_instance(1)
            time_fake.seed()
            M = random.randint(1, 50)
            target_file = '{}/log-{}.log'.format(target_path, i)
            f = open(target_file, 'w')
            for j in range(M):
                white_sentences = text_fake.sentences()
                date = str(time_fake.date_time())
                log = ' '.join([date] + white_sentences)
                f.write(log + '\n')
            f.close()

    target_file = './logs/whites'
    generate_simple_log(target_file, N)
    target_file = './logs/tests'
    generate_simple_log(target_file, 20)

def main():
    generate_simple_time_text_log()

if __name__ == '__main__':
    main()