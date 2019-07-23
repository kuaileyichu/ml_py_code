import random
import numpy as np
def mergeSort(array):
    if len(array) <= 1:#停止条件不能丢
        return array
    mid = len(array)//2
    left = array[:mid]
    right = array[mid:]
    return sort2list(mergeSort(left),mergeSort(right))

def sort2list(left,right):
    m =[]
    i=j=0
    while i<len(left) and j<len(right):
        if left[i]<=right[j]:
            m.append(left[i])
            i+=1
        else:
            m.append(right[j])
            j+=1
    if i < len(left):
        m = m + left[i:]
    if j < len(right):
        m = m + right[j:]
    return m

def quicksort(array,low,high):
    if low < high:#停止条件不能丢
        pivot = partition(array,low,high)
        quicksort(array, low, pivot-1)
        quicksort(array, pivot+1, high)

def partition(array,low,high):
    rand_index =random.randint(low,high)#random.randint(1,100)随机数中使包括1和100
    array[low], array[rand_index] = array[rand_index], array[low]
    pivot = low
    for i in range(low+1,high+1):
        if array[i]<array[low]:
            pivot+=1
            array[i],array[pivot]=array[pivot],array[i]
    array[low], array[pivot] = array[pivot], array[low]
    return pivot


def build_heap(array, size):
    for i in range(size//2,-1,-1):
        heap_adjust(array, i, size)

def heap_adjust(array, i, size):#大顶堆
    left = 2*i+1
    right = 2 * i + 2
    maxIndex = i
    if  left<size and array[left]>array[maxIndex]:
        maxIndex =left
    if  right<size and array[right]>array[maxIndex]:
        maxIndex =right
    if maxIndex !=i:
        swap(array,maxIndex,i)
        heap_adjust(array, maxIndex, size)

def heap_sorting(array):
    size = len(array)
    build_heap(array, size)
    for i in range(size-1,0,-1):
        swap(array, 0, i)
        heap_adjust(array, 0, i)



def swap(arr, i, j):
    arr[i], arr[j] = arr[j], arr[i]

if __name__ == '__main__':
    list = [34, 3, 53, 2, 1, 2, 23, 7, 14, -10]
    # list = mergeSort(list)
    # quicksort(list,0,len(list)-1)
    heap_sorting(list)
    print(list)