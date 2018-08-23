import os

def get_dataset(src_folder):
    os.chdir(src_folder)

    dataset = []
    with open('q.txt') as f:
        data = f.readlines()

    for row in data:
        row = row.replace('\n', '')
        if row is not '':
            dataset.append(row)

    return dataset

def format_board(board):
    return '{:03d}'.format(int(board))


def _format_seasons(dataset):
    new_row = []
    sel_row = []
    board1 = ''
    board2 = ''
    finish = False # для выхода во вложенном цикле из внешнего
    t = 0
    for row in dataset:
        new_row.append(row) # dataset - вся инфа из таблицы, row это сырые данные, то есть тут получаем вектор из названий эпизодов
    new_row = list(set(new_row))
    new_row = sorted(new_row)
    for i in range(0,len(new_row)):
        i=t # переходим на j-е место, так как от i до j уже запихнули в одну строку
        board1 = new_row[i][8:11]
        board2 = board1
        for j in range(i,len(new_row)):
            if new_row[i][:7] == new_row[j][:7] and not finish: # если из одного сезона идут подряд, то слепляем в строку
                temp = board2
                board2 = str(int(new_row[j][8:11])+1)
                if j+1 == len(new_row): # это для последнего класса в очереди, так как он его внешним циклом не видит
                    sel_row.append(new_row[i][:7] + '.%03d. ' + format_board(board1) + ' ' + format_board(board2)) # форматирование в виде seasons
                    finish = True # до конца дошли во вложенном if, выход из внешнего цикла
                if (int(temp) + 1) != int(board2): # проверяем на разрыв в последовательности эпизодов
                    new_board = str(int(temp))
                    sel_row.append(new_row[i][:7] + '.%03d. ' + format_board(board1) + ' ' + format_board(new_board))
                    t=j
                    break
            elif finish:
                break
            else: # если сезоны друг за другом не совпадают, а очередь еще не дошла до конца, тогда один сезон проработали и пошли дальше
                sel_row.append(new_row[i][:7] + '.%03d. ' + format_board(board1) + ' ' + format_board(board2))
                t=j
                break
    return sel_row

def create_seasons(dataset, to_write):
    os.chdir(to_write)
    with open('out.seasons', 'w') as fw:
        for row in dataset:
            fw.write(row + '\n')

dataset = get_dataset('/var/lib/mysql-files/')
seasons_dataset = _format_seasons(dataset)
create_seasons(seasons_dataset, '/home/ilya/gryaz/prj.scripts')
