#include <Python.h>
#include <omp.h>
#include <vector>
#include <set>
#include <map>
#include <string>
#include <iostream>
#include <boost/algorithm/minmax_element.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/io.hpp>

const double machine_epsilon = 1e-8;

using namespace std;
using namespace boost;
using namespace boost::numeric;

struct DATUM_X_T
{
    wstring x;
    double proba;
};
typedef vector<DATUM_X_T> DATA_X_T;
struct DATUM_XY_T
{
    wstring x;
    wstring y;
    double proba;
};
typedef vector<DATUM_XY_T> DATA_XY_T;
struct FEATURE1_T
{
    wchar_t prev_label;
    wchar_t label;
};
struct FEATURE2_T
{
    vector<int> positions;
    vector<wchar_t> values;
    wchar_t label;
};
typedef pair<vector<FEATURE1_T>, vector<FEATURE2_T>> FEATURES_T;
typedef map<wchar_t, size_t> YSET_T;

struct X_MS_ITEM_T
{
    wstring x;
    double proba;
    vector<ublas::matrix<double>> ms;
    ublas::matrix<double> alpha;
    ublas::matrix<double> beta;
    double logz;
};

inline double logsumexp(ublas::vector<double>& x )
{
    double maxval = *max_element(x.begin(), x.end());
    double val = 0;
    for(ublas::vector<double>::iterator it=x.begin(); it<x.end(); ++it)
        val += exp(*it - maxval);
    val = maxval + log(val);
    return val;
}

YSET_T pyyset_as_cpp(PyObject* pyyset)
{
    YSET_T yset;
    PyObject *key, *value;
    Py_ssize_t pos = 0;
    while (PyDict_Next(pyyset, &pos, &key, &value))
    {
        wchar_t c;
        PyUnicode_AsWideChar(key, &c, 1);
        size_t i = PyLong_AsSize_t(value);
        yset.insert(make_pair(c, i));
    }
    return yset;
}

DATA_X_T pydata_x_as_cpp(PyObject* pydata_x)
{
    DATA_X_T result;
    for(Py_ssize_t i=0, size = PyList_Size(pydata_x); i<size; ++i)
    {
        PyObject* pyxp = PyList_GetItem(pydata_x, i);
        wstring x( PyUnicode_AsUnicode( PyTuple_GetItem( pyxp, 0) ) );
        double proba = PyFloat_AsDouble( PyTuple_GetItem( pyxp, 1) );
        result.push_back({x, proba});
    }
    return result;
}

DATA_XY_T pydata_xy_as_cpp(PyObject* pydata_xy)
{
    DATA_XY_T result;
    for(Py_ssize_t i=0, size=PyList_Size(pydata_xy); i<size; ++i)
    {
        PyObject* pyxyp = PyList_GetItem( pydata_xy, i );
        PyObject* pyxy = PyTuple_GetItem( pyxyp, 0 );
        wstring x( PyUnicode_AsUnicode( PyTuple_GetItem( pyxy, 0 ) ) );
        wstring y( PyUnicode_AsUnicode( PyTuple_GetItem( pyxy, 1 ) ) );
        double proba = PyFloat_AsDouble( PyTuple_GetItem( pyxyp, 1 ) );
        assert( x.length() == y.length() );
        //wcout << x << y << proba << endl;
        result.push_back({x, y, proba});
    }
    return result;
}

vector<ublas::matrix<double>> xmatrices(const YSET_T& yset,const FEATURES_T& features,const ublas::vector<double>& weights,const wstring& x);
ublas::matrix<double> xalphas(const vector<ublas::matrix<double>>& ms);
ublas::matrix<double> xbetas(const vector<ublas::matrix<double>>& ms);

FEATURES_T pyfeatures_as_cpp(PyObject* pyfeatures)
{
    vector<FEATURE1_T> features1;
    vector<FEATURE2_T> features2;
    Py_ssize_t K = PyList_Size(pyfeatures);
    for(Py_ssize_t k=0; k<K; ++k)
    {
        PyObject* pyfeature = PyList_GetItem(pyfeatures,k);
        Py_ssize_t t = PyTuple_Size(pyfeature);
        if(t==2)
        {
            wchar_t prev_label = *PyUnicode_AsUnicode(PyTuple_GetItem(pyfeature, 0));
            wchar_t label = *PyUnicode_AsUnicode(PyTuple_GetItem(pyfeature, 1));
            features1.push_back({prev_label, label});
        }
        else if(t==3)
        {
            wchar_t label = *PyUnicode_AsUnicode(PyTuple_GetItem(pyfeature, 2));
            PyObject* pytemp = PyTuple_GetItem(pyfeature, 0);
            PyObject* pyValue = PyTuple_GetItem(pyfeature, 1);
            assert( PyTuple_Size(pytemp) == PyTuple_Size(pyValue) );
            vector<int> positions;
            vector<wchar_t> values;
            for(Py_ssize_t i=0; i<PyTuple_Size(pytemp); ++i)
            {
                int pos = PyLong_AsLong(PyTuple_GetItem(pytemp, i));
                wchar_t value = *PyUnicode_AsUnicode(PyTuple_GetItem(pyValue, i));
                positions.push_back(pos);
                values.push_back(value);
            }
            features2.push_back({positions, values, label});
        }
        else
        {
            cout << "feature size error! " << t << endl;
            exit(-1);
        }
    }
    return make_pair(features1, features2);
}

inline bool fkyyxi(const FEATURES_T& features, const size_t k,const wchar_t prev_label,const wchar_t label,const wstring& x,const size_t i)
{
    size_t K1 = features.first.size();
    if ( k<K1 )
        return features.first[k].prev_label == prev_label and features.first[k].label == label ? 1 : 0;
    if( features.second[k-K1].label == label )
    {
        size_t k3 = k-K1, n = x.length();
        for(size_t t=0, T=features.second[k3].positions.size(); t<T; ++t)
        {
            int pos = (int)i+features.second[k3].positions[t];
            if(((pos>=0 && pos<(int)n) ? x[pos] : L'\0') != features.second[k3].values[t]) return 0;
        }
        return 1;
    }
    return 0;
}

inline double featurekxy(const FEATURES_T& features,const  size_t k, const wstring& x,const wstring& y)
{
    assert( x.length() == y.length() );
    double value = 0.0;
    wchar_t prev_label = L'^';
    for(size_t i=0, n = x.length(); i<=n; ++i)
    {
        wchar_t label = i < n? y.at(i) : L'$';
        value += fkyyxi(features, k, prev_label, label, x, i);
        prev_label = label;
    }
    return value;
}

ublas::vector<double> stat_all_xy_features_values(const FEATURES_T& features, const DATA_XY_T& data_xy)
{
    size_t K1 = features.first.size();
    size_t K2 = features.second.size();
    ublas::vector<double> values(K1+K2, 0);
    #pragma omp parallel for
    for(size_t k=0; k<K1+K2; ++k)
    {
        double value = 0;
        for(DATA_XY_T::const_iterator it=data_xy.begin(); it<data_xy.end(); ++it)
            value += it->proba * featurekxy(features, k, it->x, it->y);
        values(k) = value;
    }
    return values;
}

ublas::vector<double> doublelist_as_cpp(PyObject* pydoublelist)
{
    ublas::vector<double> vec(PyList_Size(pydoublelist));
    for(Py_ssize_t i=0; i<PyList_Size(pydoublelist); ++i)
    {
        vec(i) = PyFloat_AsDouble(PyList_GetItem(pydoublelist, i));
    }
    return vec;
}

static PyObject* stat_all_xy_features_values(PyObject* self, PyObject* args)
{
    PyObject* pyfeatures = NULL;
    PyObject* pydata_xy  = NULL;
    if (!PyArg_ParseTuple(args, "OO", &pyfeatures, &pydata_xy))
        return NULL;
    FEATURES_T features = pyfeatures_as_cpp(pyfeatures);
    DATA_XY_T data_xy = pydata_xy_as_cpp(pydata_xy);
    ublas::vector<double> values = stat_all_xy_features_values(features, data_xy);
    ublas::vector<double>::size_type k=0, K=values.size();
    PyObject* pyvalues = PyList_New(K);
    for(; k<K; ++k) PyList_SetItem(pyvalues, k, PyFloat_FromDouble( values(k) ) );
    return pyvalues;
}

double model_values(const ublas::vector<double>& weights,const YSET_T& yset,const FEATURES_T& features, const DATA_X_T& data_x)
{
    double val = 0;
    #pragma omp parallel for reduction(+:val)
    for( DATA_X_T::const_iterator it=data_x.begin(); it<data_x.end(); ++it )
    {
        vector<ublas::matrix<double>> ms = xmatrices(yset, features, weights, it->x);
        ublas::matrix<double> alpha = xalphas(ms);
        ublas::matrix<double> beta = xbetas(ms);
        ublas::vector<double> alphalastrow = ublas::matrix_row<ublas::matrix<double>>(alpha, alpha.size1()-1);
        ublas::vector<double> betalastrow = ublas::matrix_row<ublas::matrix<double>>(beta, 0);
        double logz1 = logsumexp(alphalastrow);
        double logz2 = logsumexp(betalastrow);
        //cout << logz1 << " , " << logz2 << " abs:" << abs(logz1 - logz2) << endl;
        assert( abs(logz1 - logz2) < machine_epsilon );
        val += it->proba * logz1;
    }
    return val;
}

static PyObject* model_values(PyObject* self, PyObject* args)
{
    PyObject* pyweights = NULL;
    PyObject* pyyset = NULL;
    PyObject* pyfeatures = NULL;
    PyObject* pydata_x  = NULL;
    if (!PyArg_ParseTuple(args, "OOOO", &pyweights, &pyyset, &pyfeatures, &pydata_x))
        return NULL;
    ublas::vector<double> weights = doublelist_as_cpp(pyweights);
    YSET_T yset = pyyset_as_cpp(pyyset);
    FEATURES_T features = pyfeatures_as_cpp(pyfeatures);
    DATA_X_T data_x = pydata_x_as_cpp(pydata_x);
    return PyFloat_FromDouble(model_values(weights, yset, features, data_x));
}

inline double yyxi(const FEATURES_T& features,const ublas::vector<double>& weights,const wchar_t prev_label,const wchar_t label,const wstring& x,const size_t i)
{
    static map<tuple<wchar_t, wchar_t, wstring, size_t>, vector<size_t>> yyxi_cache_helper;
    double val=0;
    auto key = make_tuple(prev_label, label, x, i);
    if( yyxi_cache_helper.find(key) != yyxi_cache_helper.end() )
    {
        vector<size_t> ks = yyxi_cache_helper[key];
        for(vector<size_t>::iterator it=ks.begin(); it<ks.end(); ++it)
            val += weights[*it];
        return val;
    }
    for(size_t k=0; k<weights.size(); ++k)
    {
        if( fkyyxi(features, k, prev_label, label, x, i) > 0.5 )
        {
            if ( yyxi_cache_helper.find(key) == yyxi_cache_helper.end() )
            {
                #pragma omp critical
                yyxi_cache_helper.insert(make_pair(key, vector<size_t>()));
            }
            #pragma omp critical
            yyxi_cache_helper[key].push_back(k);
            val += weights[k];
        }
    }
    return val;
}

inline ublas::matrix<double> xmatrixi(const YSET_T& yset,const FEATURES_T& features,const ublas::vector<double>& weights,const wstring& x,const size_t i)
{
    ublas::matrix<double> m(yset.size(), yset.size());
    for(YSET_T::const_iterator y0it=yset.begin(); y0it!=yset.end(); ++y0it)
    {
        for(YSET_T::const_iterator y1it=yset.begin(); y1it!=yset.end(); ++y1it)
        {
            if(i==0)
            {
                m(y0it->second, y1it->second) = (y0it->second==0) ? yyxi(features, weights, L'^', y1it->first, x, i) : 0;
            }
            else if(i==x.length())
            {
                m(y0it->second, y1it->second) = (y1it->second==0) ? yyxi(features, weights, y0it->first, L'$', x, i) : 0;
            }
            else
            {
                m(y0it->second, y1it->second) = yyxi(features, weights, y0it->first, y1it->first, x, i);
            }
        }
    }
    return m;
}

inline vector<ublas::matrix<double>> xmatrices(const YSET_T& yset,const FEATURES_T& features,const ublas::vector<double>& weights,const wstring& x)
{
    size_t n = x.length();
    vector<ublas::matrix<double>> ms(n+1);
    for(size_t i=0; i<=n; ++i)
        ms.at(i) = xmatrixi(yset, features, weights, x, i);
    return ms;
}

inline ublas::matrix<double> xalphas(const vector<ublas::matrix<double>>& ms)
{
    size_t mslen = ms.size();
    ublas::matrix<double> alpha(mslen, ms[0].size1());
    ublas::row<ublas::matrix<double>>(alpha, 0) = ublas::row<ublas::matrix<double>>(ms[0], 0);
    for(size_t i=1; i<mslen-1; ++i)
    {
        for(size_t j=0; j<ms[i].size2(); ++j)
        {
            ublas::vector<double> values = ublas::row<ublas::matrix<double>>(alpha, i-1)
                                           + ublas::column<ublas::matrix<double>>(ms[i], j);
            alpha(i, j) = logsumexp( values );
        }
    }
    ublas::row<ublas::matrix<double>>(alpha, mslen-1) = ublas::row<ublas::matrix<double>>(alpha, mslen-2)
                                   + ublas::column<ublas::matrix<double>>(ms[mslen-1], 0);
    return alpha;
}

inline ublas::matrix<double> xbetas(const vector<ublas::matrix<double>>& ms)
{
    size_t mslen = ms.size();
    ublas::matrix<double> beta(mslen, ms[0].size1());
    ublas::row<ublas::matrix<double>>(beta, mslen-1) = ublas::column<ublas::matrix<double>>(ms[mslen-1], 0);
    for(size_t i=mslen-2; i>0; --i)
    {
        for(size_t j=0; j<ms[i].size1(); ++j)
        {
            ublas::vector<double> values = ublas::row<ublas::matrix<double>>(ms[i], j)
                                           + ublas::row<ublas::matrix<double>>(beta, i+1);
            beta(i, j) = logsumexp( values );
        }
    }
    ublas::row<ublas::matrix<double>>(beta, 0) = ublas::row<ublas::matrix<double>>(ms[0], 0)
                                   + ublas::row<ublas::matrix<double>>(beta, 1);
    return beta;
}

inline double xyiiproba(const YSET_T& yset,const vector<ublas::matrix<double>>& ms,const ublas::matrix<double>& alpha,const ublas::matrix<double>& beta,const size_t s0,const size_t s1,const size_t i,const double logz)
{
    //assert( 0<=i and i<ms.size() );
    double p = -logz;
    if (i==0)
    {
        p += ms[i](0, s1) + beta(i+1, s1);
    }
    else if(i==ms.size()-1)
    {
        p += alpha(i-1, s0) + ms[i](s0, 0);
    }
    else
    {
        p += alpha(i-1, s0) + ms[i](s0, s1) + beta(i+1, s1);
    }
    return exp( p );
}

inline double xyproba(const YSET_T& yset, const vector<ublas::matrix<double>>& ms, const double logz,const wstring& y)
{
    double v=0;
    v += ms.front()(0, yset.at(y.front()));
    for(size_t i=1; i<y.size(); ++i)
        v+=ms.at(i)(yset.at(y.at(i-1)), yset.at(y.at(i)));
    v += ms.back()( yset.at(y.back()), 0);
    return exp( v - logz );
}

inline double kx(const YSET_T& yset,const FEATURES_T& features,const size_t k,const wstring& x,const vector<ublas::matrix<double>>& ms,const ublas::matrix<double>& alpha,const ublas::matrix<double>& beta,const double logz)
{
    //assert( x.length()+1 == ms.size() );
    double v = 0;
    size_t n = x.length();

    for(YSET_T::const_iterator y1it=yset.begin(); y1it!=yset.end(); ++y1it)
        if( fkyyxi(features, k, L'^', y1it->first, x, 0) )
            v += xyiiproba(yset, ms, alpha, beta, 0, y1it->second, 0, logz);

    for(size_t i=1; i<n; ++i)
        for(YSET_T::const_iterator y0it=yset.begin(); y0it!=yset.end(); ++y0it)
            for(YSET_T::const_iterator y1it=yset.begin(); y1it!=yset.end(); ++y1it)
                if( fkyyxi(features, k, y0it->first, y1it->first, x, i) )
                    v += xyiiproba(yset, ms, alpha, beta, y0it->second, y1it->second, i, logz);

    for(YSET_T::const_iterator y0it=yset.begin(); y0it!=yset.end(); ++y0it)
        if( fkyyxi(features, k, y0it->first, L'$', x, n) )
            v += xyiiproba(yset, ms, alpha, beta, y0it->second, 0, n, logz);

    return v;
}

inline double allxv(const YSET_T& yset, const FEATURES_T& features,const size_t k,const vector<X_MS_ITEM_T>& malphabetaz)
{
    double grad=0;
    for(vector<X_MS_ITEM_T>::const_iterator it=malphabetaz.cbegin(); it<malphabetaz.cend(); ++it)
        grad += it->proba * kx( yset, features, k, it->x, it->ms, it->alpha, it->beta, it->logz );
    return grad;
}

ublas::vector<double> model_grads(const YSET_T& yset,const FEATURES_T& features,const ublas::vector<double>& weights,const DATA_X_T& data_x)
{
    size_t K = weights.size();
    ublas::vector<double> grads(K);
    vector<X_MS_ITEM_T> malphabetaz( data_x.size() );
    #pragma omp parallel for
    for(DATA_X_T::size_type i=0; i<data_x.size(); ++i)
    {
        vector<ublas::matrix<double>> ms = xmatrices( yset, features, weights, data_x.at(i).x );
        ublas::matrix<double> alpha = xalphas(ms);
        ublas::matrix<double> beta = xbetas(ms);
        ublas::vector<double> alphalastrow = ublas::matrix_row<ublas::matrix<double>>(alpha, alpha.size1()-1);
        ublas::vector<double> betafirstrow = ublas::matrix_row<ublas::matrix<double>>(beta, 0);
        double logz1 = logsumexp( alphalastrow );
        double logz2 = logsumexp( betafirstrow );
        assert( abs(logz1 - logz2) < machine_epsilon );
        assert( data_x.at(i).x.length()+1 == ms.size() );
        malphabetaz.at(i) = {data_x.at(i).x, data_x.at(i).proba, ms, alpha, beta, logz1};
    }
    #pragma omp parallel for
    for(size_t k=0; k<K; ++k)
        grads(k) = allxv(yset, features, k, malphabetaz);
    return grads;
}

ublas::vector<double> model_grads1(const YSET_T& yset,const FEATURES_T& features,const ublas::vector<double>& weights,const DATA_X_T& data_x)
{
    size_t K = weights.size();
    ublas::vector<double> grads(K);
    vector<X_MS_ITEM_T> malphabetaz( data_x.size() );
    #pragma omp parallel for
    for(DATA_X_T::size_type i=0; i<data_x.size(); ++i)
    {
        vector<ublas::matrix<double>> ms = xmatrices( yset, features, weights, data_x.at(i).x );
        ublas::matrix<double> alpha = xalphas(ms);
        ublas::matrix<double> beta = xbetas(ms);
        ublas::vector<double> alphalastrow = ublas::matrix_row<ublas::matrix<double>>(alpha, alpha.size1()-1);
        ublas::vector<double> betafirstrow = ublas::matrix_row<ublas::matrix<double>>(beta, 0);
        double logz1 = logsumexp( alphalastrow );
        double logz2 = logsumexp( betafirstrow );
        assert( abs(logz1 - logz2) < machine_epsilon );
        assert( data_x.at(i).x.length()+1 == ms.size() );
        malphabetaz.at(i) = {data_x.at(i).x, data_x.at(i).proba, ms, alpha, beta, logz1};
    }
    struct XS0S1I_T
    {
        size_t s0;
        size_t s1;
        size_t xi;
    };
    typedef vector<XS0S1I_T> XS0S1I_LIST_T;
    struct KX_T
    {
        size_t xnum;
        XS0S1I_LIST_T pxs;
    };
    typedef vector<KX_T> KX_LIST_T;
    static vector<KX_LIST_T> kxtablecache(K);
    static bool kxtablecache_flag = 0;
    if ( !kxtablecache_flag )
    {
        #pragma omp parallel for
        for(size_t k=0; k<K; ++k)
        {
            KX_LIST_T kx;
            for(size_t xnum=0; xnum<data_x.size(); ++xnum)
            {
                XS0S1I_LIST_T xs0s1i;
                wstring x = data_x.at(xnum).x;
                size_t n = x.length();
                for(YSET_T::const_iterator y1it=yset.begin(); y1it!=yset.end(); ++y1it)
                    if( fkyyxi(features, k, L'^', y1it->first, x, 0) )
                        xs0s1i.push_back({ 0, y1it->second, 0 });
                for(size_t i=1; i<n; ++i)
                    for(YSET_T::const_iterator y0it=yset.begin(); y0it!=yset.end(); ++y0it)
                        for(YSET_T::const_iterator y1it=yset.begin(); y1it!=yset.end(); ++y1it)
                            if( fkyyxi(features, k, y0it->first, y1it->first, x, i) )
                                xs0s1i.push_back({ y0it->second, y1it->second, i });
                for(YSET_T::const_iterator y0it=yset.begin(); y0it!=yset.end(); ++y0it)
                    if( fkyyxi(features, k, y0it->first, L'$', x, n) )
                        xs0s1i.push_back( {y0it->second, 0, n});
                if ( xs0s1i.size() > 0 )
                    kx.push_back({xnum, xs0s1i});
            }
            kxtablecache.at(k) = kx;
        }
        kxtablecache_flag = 1;
    }

    #pragma omp parallel for
    for(size_t k=0; k<K; ++k)
    {
        const KX_LIST_T& kx = kxtablecache.at(k);
        for( KX_LIST_T::const_iterator kxit=kx.begin(); kxit<kx.end(); ++kxit  )
        {
            const X_MS_ITEM_T& xitem = malphabetaz.at(kxit->xnum);
            const XS0S1I_LIST_T& xs0s1i = kxit->pxs;
            double v=0;
            for( XS0S1I_LIST_T::const_iterator it=xs0s1i.begin(); it<xs0s1i.end(); ++it )
            {
                v += xyiiproba(yset, xitem.ms, xitem.alpha, xitem.beta, it->s0, it->s1, it->xi, xitem.logz );
            }
            grads(k) += xitem.proba * v;
        }
    }
    return grads;
}

static PyObject* model_grads(PyObject* self, PyObject* args)
{
    PyObject* pyyset = NULL;
    PyObject* pyfeatures = NULL;
    PyObject* pyweights = NULL;
    PyObject* pydata_x  = NULL;
    if (!PyArg_ParseTuple(args, "OOOO", &pyyset, &pyfeatures, &pyweights, &pydata_x))
        return NULL;
    YSET_T yset = pyyset_as_cpp(pyyset);
    FEATURES_T features = pyfeatures_as_cpp(pyfeatures);
    ublas::vector<double> weights = doublelist_as_cpp(pyweights);
    DATA_X_T data_x = pydata_x_as_cpp(pydata_x);
    ublas::vector<double> grads = model_grads(yset, features, weights, data_x);
    PyObject* pygrads = PyList_New(grads.size());
    for(size_t i=0; i<grads.size(); ++i)
        PyList_SetItem(pygrads, i, PyFloat_FromDouble( grads(i) ) );
    return pygrads;
}

static PyObject* model_grads1(PyObject* self, PyObject* args)
{
    PyObject* pyyset = NULL;
    PyObject* pyfeatures = NULL;
    PyObject* pyweights = NULL;
    PyObject* pydata_x  = NULL;
    if (!PyArg_ParseTuple(args, "OOOO", &pyyset, &pyfeatures, &pyweights, &pydata_x))
        return NULL;
    YSET_T yset = pyyset_as_cpp(pyyset);
    FEATURES_T features = pyfeatures_as_cpp(pyfeatures);
    ublas::vector<double> weights = doublelist_as_cpp(pyweights);
    DATA_X_T data_x = pydata_x_as_cpp(pydata_x);
    ublas::vector<double> grads = model_grads1(yset, features, weights, data_x);
    PyObject* pygrads = PyList_New(grads.size());
    for(size_t i=0; i<grads.size(); ++i)
        PyList_SetItem(pygrads, i, PyFloat_FromDouble( grads(i) ) );
    return pygrads;
}

static PyMethodDef crfextMethods[] =
{
    {"model_grads", model_grads, METH_VARARGS, "grads"},
    {"model_grads1", model_grads1, METH_VARARGS, "grads"},
    {"model_values", model_values, METH_VARARGS, ""},
    {"stat_all_xy_features_values", stat_all_xy_features_values, METH_VARARGS, "stat all features values on all data"},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

static struct PyModuleDef crfextmodule =
{
    PyModuleDef_HEAD_INIT,
    "crfext",
    NULL,
    -1,
    crfextMethods
};

PyMODINIT_FUNC PyInit_crfext(void)
{
    return PyModule_Create(&crfextmodule);
};
