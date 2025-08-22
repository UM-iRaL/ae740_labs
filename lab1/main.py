from random_vector import RandomVector
def main():
    # Create a RandomVector of size 20
    rv = RandomVector(20)
    # Print all values
    rv.print()
    # Print statistics
    print(f"Mean: {rv.mean():.6f}")
    print(f"Min: {rv.min():.6f}")
    print(f"Max: {rv.max():.6f}")
    # Print histogram with 8 bins
    print("Histogram:")
    rv.print_histogram(8)
if __name__ == "__main__":
    main()