
class IndexIterator<S> extends Iterator<int> {
  int? _currentIndex;
  final Iterator<S> _iterator;
  final bool Function(S) _f;

  IndexIterator(this._iterator, this._f) : _currentIndex = -1;

  bool moveNext() {
    if (_currentIndex != null) {
      while (_iterator.moveNext()) {
        _currentIndex = _currentIndex! + 1;
        if (_f(_iterator.current)) return true;
      }
      _currentIndex = null;
    }
    return false;
  }

  int get current => _currentIndex as int;
}

class IndexIterable<S> extends Iterable<int> {
  final Iterable<S> _iterable;
  final bool Function(S) _f;

  factory IndexIterable(Iterable<S> iterable, bool function(S value)) {
    return new IndexIterable<S>._(iterable, function);
  }

  IndexIterable._(this._iterable, this._f);

  Iterator<int> get iterator => new IndexIterator<S>(_iterable.iterator, _f);
}

extension IterableIndices<E> on Iterable<E> {
  Iterable<int> indicesWhere(bool Function(E) f) => IndexIterable<E>(this, f);
}
